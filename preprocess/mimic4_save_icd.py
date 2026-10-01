# script adjusted to MIMIC-IV by Bernie Chen

# This script processes MIMIC-III dataset and builds a binary matrix or a count matrix depending on your input.
# The output matrix is a Numpy matrix of type float32, and suitable for training medGAN.
# Written by Edward Choi (mp2893@gatech.edu)
# Usage: Put this script to the folder where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output file> <"binary"|"count">
# Note that the last argument "binary/count" determines whether you want to create a binary matrix or a count matrix.

# Output files
# <output file>.pids: cPickled Python list of unique Patient IDs. Used for intermediate processing
# <output file>.matrix: Numpy float32 matrix. Each row corresponds to a patient. Each column corresponds to a ICD9 diagnosis code.
# <output file>.types: cPickled Python dictionary that maps string diagnosis codes to integer diagnosis codes.

import sys
import _pickle as pickle
import numpy as np
from datetime import datetime
import pandas as pd

def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr
    
def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3]
        else: return dxStr

if __name__ == '__main__':
    admissionFile = sys.argv[1]
    diagnosisFile = sys.argv[2]
    outFile = sys.argv[3]

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[0])
        admId = int(tokens[1])
        admTime = datetime.strptime(tokens[2], '"%Y-%m-%d %H:%M:%S"')
        admDateMap[admId] = admTime
        if pid in pidAdmMap: pidAdmMap[pid].append(admId)
        else: pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-dxList mapping')
    admDxMap = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        # admId = int(tokens[2])
        admId = int(tokens[1])
        dxStr = 'ICD' + str(tokens[4]) + ": " + str(tokens[3]).strip()
        # dxStr = 'D_' + convert_to_icd9(tokens[3][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
        # dxStr = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])
        if admId in admDxMap: admDxMap[admId].append(dxStr)
        else: admDxMap[admId] = [dxStr]
    infd.close()

    print('Building pid-sortedVisits mapping')
    
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        #if len(admIdList) < 2: continue
        
        # !! Not all admissionIDs have icd codes, so need to account
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList if admId in admDxMap])
        pidSeqMap[pid] = sortedList


    
    
    print('Building pids, dates, strSeqs')
    pids = []
    dates = []
    seqs = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
        dates.append(date)
        seqs.append(seq)
    
    # seqs is list of patient, visits, and their icd codes

    print('Converting strSeqs to intSeqs, and making types')
    types = {} # numerating icd codes
    code_counts = {} 
    newSeqs = []
    icd_codes_in_order = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                    code_counts[code] += 1
                else:
                    types[code] = len(types)
                    code_counts[code] = 1
                    newVisit.append(types[code])
                    icd_codes_in_order.append(code)
            newPatient.append(newVisit)
        newSeqs.append(newPatient)
    # sorted from most frequent icd code to least frequent
    code_counts = dict(sorted(code_counts.items(), key=lambda item: item[1], reverse=True))
    num_codes = len(code_counts)
    
    print(code_counts)
    code_index = {} # mapping icd code to index
    ind = 0
    for key, item in code_counts.items():
        code_index[key] = ind
        ind+=1
        
    print('Constructing the matrix')
    numPatients = len(seqs)
    matrix = np.zeros((numPatients, num_codes)).astype('float32')
    for i, patient in enumerate(seqs):
        for visit in patient:
            for code in visit:
                matrix[i][code_index[code]] = 1

    
    print("Constructing dictionary")
    patient_to_index = {}
    for i, patient_id in enumerate(pids):
        patient_to_index[patient_id] = i

    
    # print(patient_to_index)
    # print(len(matrix))
    
    # pickle.dump(pids, open(outFile+'.pids', 'wb'), -1)
    pickle.dump(matrix, open(outFile+'.matrix', 'wb'), -1)
    # pickle.dump(types, open(outFile+'.types', 'wb'), -1)

    pickle.dump(icd_codes_in_order, open("icd_codes_in_order"+'.list', 'wb'), -1)
    pickle.dump(pids, open("patient_ids_in_order"+'.list', 'wb'), -1) 
    pickle.dump(code_counts, open("code_freqs"+".dict", 'wb'), -1)
    pickle.dump(patient_to_index, open("patient_to_index"+".dict", 'wb'), -1)