"""MIDST MIA -- fast, numerically equivalent MixedDiffusion loss + CUDA-graph training.

MixedDiffusion.forward() launches ~600 tiny kernels per micro-batch (per-class
slicing, one_hot, in-place index_put), which makes a 200k-parameter LSTM cost
10 ms/step.  Every categorical variable in TimeDiff's EHR configs is binary, so
the multinomial part vectorises as (B, n_cat, 2, T).  `check()` verifies each
term against the original implementation.

Gradient accumulation of 2 micro-batches of 32 with loss/2 each equals one batch
of 64 (every term is a batch mean), so one graph step = one ETDiff optimiser step.

Verify on a model and a batch with:  check(diffusion, x, num_idx, cat_idx)
"""
import math, time
import torch
from ema_pytorch import EMA


def _lae(a, b):  # log_add_exp
    m = torch.maximum(a, b)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m))


class FastMixedLoss:
    def __init__(self, diff, num_idx, cat_idx):
        self.d = diff
        self.num_idx = torch.tensor(num_idx, device=diff.betas.device)
        self.cat_idx = torch.tensor(cat_idx, device=diff.betas.device)
        self.nc = len(cat_idx)
        self.T = diff.num_timesteps
        self.log2 = math.log(2.0)

    def _g(self, buf, t):  # gather per-sample scalar -> (B,1,1,1)
        return buf.gather(0, t).view(-1, 1, 1, 1)

    def q_pred(self, logx0, t):  # (B,nc,2,T): log q(x_t | x_0)
        return _lae(logx0 + self._g(self.d.log_cumprod_alpha, t),
                    self._g(self.d.log_1_min_cumprod_alpha, t) - self.log2)

    def q_pred_one(self, logxt, t):
        return _lae(logxt + self._g(self.d.log_alpha, t), self._g(self.d.log_1_min_alpha, t) - self.log2)

    def q_posterior(self, logx0, logxt, t):
        tm1 = (t - 1).clamp(min=0)
        ev = self.q_pred(logx0, tm1)
        ev = torch.where((t == 0).view(-1, 1, 1, 1), logx0, ev)
        un = ev + self.q_pred_one(logxt, t)
        return un - torch.logsumexp(un, dim=2, keepdim=True)

    @staticmethod
    def log_onehot(c):  # c: (B,nc,T) in {0,1} -> (B,nc,2,T)
        oh = torch.stack([1 - c, c], dim=2)
        return torch.log(oh.clamp(min=1e-30))

    def __call__(self, x, t=None, noise=None, gumbel_u=None, return_parts=False):
        d = self.d
        B, _, L = x.shape
        if t is None:
            t = torch.randint(0, self.T, (B,), device=x.device)
        xn = x.index_select(1, self.num_idx) * 2 - 1
        if noise is None:
            noise = torch.randn_like(xn)
        a = d.sqrt_alphas_cumprod.gather(0, t).view(-1, 1, 1)
        s = d.sqrt_one_minus_alphas_cumprod.gather(0, t).view(-1, 1, 1)
        xt = a * xn + s * noise
        c = x.index_select(1, self.cat_idx)
        logx0 = self.log_onehot(c)
        lq = self.q_pred(logx0, t)
        if gumbel_u is None:
            gumbel_u = torch.rand_like(lq)
        gmb = -torch.log(-torch.log(gumbel_u + 1e-30) + 1e-30)
        k = (gmb + lq).argmax(dim=2).float()
        logxt = self.log_onehot(k)
        out = d.model(torch.cat([xt, logxt.reshape(B, 2 * self.nc, L)], 1), t, None)
        n_num = xn.shape[1]
        out_num, out_cat = out[:, :n_num], out[:, n_num:].reshape(B, self.nc, 2, L)
        gauss = ((out_num - noise) ** 2).mean(dim=(1, 2))                   # loss_weight == 1
        logx0_hat = torch.log_softmax(out_cat, dim=2)
        log_true = self.q_posterior(logx0, logxt, t)
        log_model = self.q_posterior(logx0_hat, logxt, t)
        kl = (log_true.exp() * (log_true - log_model)).sum(dim=(1, 2, 3))
        nll = -(logx0.exp() * log_model).sum(dim=(1, 2, 3))
        Lt = torch.where(t == 0, nll, kl)
        tT = torch.full_like(t, self.T - 1)
        lqT = self.q_pred(logx0, tT)
        kl_prior = (lqT.exp() * (lqT + self.log2)).sum(dim=(1, 2, 3))
        cat = (Lt * self.T + kl_prior) / self.nc                           # kl / pt, pt = 1/T
        if return_parts:
            return gauss, cat, dict(t=t, noise=noise, logxt=logxt, out=out)
        return d.loss_lambda * cat.mean() + gauss.mean()


def check(diff, x, num_idx, cat_idx):
    """Compare every term with MixedDiffusion's own functions."""
    from models.ETDiff.mixed_diffusion import index_to_log_onehot
    fl = FastMixedLoss(diff, num_idx, cat_idx)
    with torch.no_grad():
        g, c, p = fl(x, return_parts=True)
        t, noise, logxt, out = p["t"], p["noise"], p["logxt"], p["out"]
        B, L = x.shape[0], x.shape[2]
        xc = x[:, cat_idx]
        lx = index_to_log_onehot(xc.long(), diff.categorical_num_classes)
        lxt = logxt.reshape(B, -1, L)
        out_cat = out[:, len(num_idx):]
        pt = torch.ones_like(t).float() / diff.num_timesteps
        c_ref = diff.cat_loss(out_cat, lx, lxt, t, pt, None) / len(cat_idx)
        g_ref = ((out[:, :len(num_idx)] - noise) ** 2).mean(dim=(1, 2))
        # the model input the original would have built from the same draws
        xn = diff.normalize(x[:, num_idx])
        xt_ref = diff.gauss_q_sample(xn, t, noise=noise)
        out_ref = diff.model(torch.cat([xt_ref, lxt], 1), t, None)
    return dict(gauss_maxabs=float((g - g_ref).abs().max()), cat_maxrel=float(((c - c_ref).abs() / c_ref.abs().clamp(min=1e-6)).max()),
                model_out_maxabs=float((out - out_ref).abs().max()))


def train_graphed(diff, X, steps, num_idx, cat_idx, batch=64, lr=8e-5, ema_decay=0.995,
                  ema_every=10, seed=0, log=None, ckpt_every=None, ckpt_fn=None):
    """ETDiff.train() as one CUDA graph per optimiser step.  X: (N, C, L) on GPU, [0,1]."""
    dev = X.device
    torch.manual_seed(seed)
    diff = diff.to(dev).train()
    fl = FastMixedLoss(diff, num_idx, cat_idx)
    opt = torch.optim.Adam(diff.parameters(), lr=lr, betas=(0.9, 0.99), capturable=True)
    # The EMA object is always built and updated, even when the caller wants raw weights:
    # that is the code path validated for 700k-step runs (the no-EMA path faulted twice).
    ema = EMA(diff, beta=ema_decay or 0.995, update_every=ema_every).to(dev)
    return_ema = bool(ema_decay)
    params = [p for p in diff.parameters()]
    n = len(X)
    sx = torch.empty((batch,) + tuple(X.shape[1:]), device=dev)

    def step():
        loss = fl(sx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
        opt.step()
        return loss

    # epoch-wise reshuffling like a DataLoader(shuffle=True, drop_last) would do
    g = torch.Generator(device=dev); g.manual_seed(seed)
    perm, pos = torch.randperm(n, device=dev, generator=g), 0

    def load_batch():
        nonlocal perm, pos
        if pos + batch > n:
            perm, pos = torch.randperm(n, device=dev, generator=g), 0
        sx.copy_(X[perm[pos:pos + batch]]); pos += batch

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            load_batch(); opt.zero_grad(set_to_none=True); step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        sloss = step()
    t0 = time.time()
    for it in range(1, steps + 1):
        load_batch()
        graph.replay()
        ema.update()
        if log and it % 20000 == 0:
            print(f"[{log}] step {it}/{steps} loss {sloss.item():.4f} {(time.time()-t0)/it*1e3:.2f} ms/step", flush=True)
        if ckpt_every and it % ckpt_every == 0 and ckpt_fn:
            ckpt_fn(it, ema.ema_model if return_ema else diff)
            diff.train()
    return (ema.ema_model if return_ema else diff).eval()
