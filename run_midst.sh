#!/bin/bash
# MIDST MIA end to end, spread over the GPUs.
#
#   ./run_midst.sh                                  # MIMIC-IV, 64 shadows, GPUs 0 and 1
#   DATASET=physionet K=32 GPUS="0" ./run_midst.sh  # public stand-in, one GPU
#
# Block 0 (target) runs first. Then each shadow k runs Block 1 (base shadow) and
# Block 2 (synth-shadow) back to back, JOBS_PER_GPU at a time on every GPU, while
# the proxy (Block 4 --proxy_only) trains alongside. Blocks 3 and 4 finish.
# Every step skips work whose output already exists, so a rerun resumes.
set -e
cd "$(dirname "$0")"
DATASET=${DATASET:-mimic}
K=${K:-64}
GPUS=${GPUS:-"0 1"}
JOBS_PER_GPU=${JOBS_PER_GPU:-2}
OUT=${OUT:-block1_$DATASET}
PY=${PY:-python}
ARGS="--dataset $DATASET --output_dir $OUT"
mkdir -p "$OUT/logs"

first_gpu=${GPUS%% *}
CUDA_VISIBLE_DEVICES=$first_gpu $PY MIDST_block0_mimic.py $ARGS > "$OUT/logs/block0.log" 2>&1

CUDA_VISIBLE_DEVICES=$first_gpu $PY MIDST_block4_mimic.py $ARGS --proxy_only \
    > "$OUT/logs/proxy.log" 2>&1 &

shadow() {   # gpu k
    CUDA_VISIBLE_DEVICES=$1 $PY MIDST_block1_mimic.py $ARGS --shadows $2 > "$OUT/logs/block1_k$2.log" 2>&1 &&
    CUDA_VISIBLE_DEVICES=$1 $PY MIDST_block2_mimic.py $ARGS --shadows $2 > "$OUT/logs/block2_k$2.log" 2>&1
}
export -f shadow; export PY ARGS OUT
gpus=($GPUS)
for g in "${!gpus[@]}"; do
    seq $((g + 1)) ${#gpus[@]} $K | xargs -P $JOBS_PER_GPU -I{} bash -c "shadow ${gpus[$g]} {}" &
done
wait

$PY MIDST_block3_mimic.py $ARGS > "$OUT/logs/block3.log" 2>&1
CUDA_VISIBLE_DEVICES=$first_gpu $PY MIDST_block4_mimic.py $ARGS > "$OUT/logs/block4.log" 2>&1
echo "done: $OUT/result.json"
