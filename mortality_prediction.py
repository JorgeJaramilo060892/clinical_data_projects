import os
import csv
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm 

patients = pd.read_csv("/Users/jorgejaramillobermudez/Desktop/projects/mimic-iii-clinical-database-demo-1.4/PATIENTS.csv")
print(patients.shape)
print(patients.head())

# Convert date of birth = to date
patients['dob'] = pd.to_datetime(patients['dob']).dt.date

# prepare mortality level 
# Note: A valid dod_hosp means that the patient died during an individual hospital admissions or ICU stay (label 1)
patients['mortality'] = patients['dod_hosp'].apply(lambda x: 0 if x != x else 1)
patients = patients[['subject_id', 'gender', 'dob', 'mortality']]
print(patients.head())


# Admission:  This table defines a patient's hospital admission, [hadm_id]
admissions = pd.read_csv("/Users/jorgejaramillobermudez/Desktop/projects/mimic-iii-clinical-database-demo-1.4/ADMISSIONS.csv")
print(admissions.shape)
print(admissions.head())

#Convert admittime and dischtime to date
#Similar to dob, by converting them to date, we can easily calculate the patient age.

admissions['admittime'] = pd.to_datetime(admissions['admittime']).dt.date
admissions['dischtime'] = pd.to_datetime(admissions['dischtime']).dt.date
admissions = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime']]
print(admissions.shape)
admissions.head()


#Merge patient and admission info by subject_id 
patients_admissions = pd.merge(patients, admissions , on="subject_id" )

# Excludo patients whose age <18, Adults 
# calculate admissions age
patients_admissions['age'] = patients_admissions.apply(lambda x: (x['admittime'] -  x['dob']).days // 365.25, axis=1)
# for patients with age >89, set it to 89
patients_admissions['age'] = patients_admissions['age'].apply(lambda x: 89 if x > 89 else x)

print('# of patients with age < 18:', len(patients_admissions[patients_admissions['age'] < 18].groupby('hadm_id')))
print('# of patients with age >= 89:', len(patients_admissions[patients_admissions['age'] >= 89].groupby('hadm_id')))

patients_admissions = patients_admissions[patients_admissions['age'] >= 18].reset_index(drop=True)
# drop dob column
patients_admissions = patients_admissions.drop(columns=['dob'])
print(patients_admissions.head())

patients_admissions['age'] = patients_admissions['age'].astype(int) # drop .0


# Diagnosis code
#This table contains ICD diagnoses for patients, most notably ICD-9 diagnoses.
# Set of valid admissions ids
valid_adm_ids = set(patients_admissions.hadm_id)
def convert_to_3digit_idc9(dxSTR):
    '''convert ICD9 to 3-digit-version'''
    if dxSTR.startswith('E'):
        if len(dxSTR) > 4:
            return dxSTR[:4]
        else:
            return dxSTR
    else:
        if len(dxSTR) > 3:
            return dxSTR[:3]
        else:
            return dxSTR
print(patients_admissions.head())

diagnosis_icd = pd.read_csv("/Users/jorgejaramillobermudez/Desktop/projects/mimic-iii-clinical-database-demo-1.4/DIAGNOSES_ICD.csv")
print(diagnosis_icd.head())

#Drop invalid admissions
# Drop admissions not in valid_adm_ids.

print("# of rows with invalid admissions:", np.count_nonzero(diagnosis_icd['hadm_id'].isin(valid_adm_ids) == False))
diagnosis_icd = diagnosis_icd[diagnosis_icd['hadm_id'].isin(valid_adm_ids)].reset_index(drop=True)
print("Rows with invalid admissions are dropped! Shape:", diagnosis_icd.shape)


# Convert to ICD9 3-digit
#Since we only have very limited data, converting ICD9 to 3-digit version will make the learning process easier (e.g., the representation will be much smaller).
diagnosis_icd['icd9_3digit'] = diagnosis_icd['icd9_code'].apply(lambda x: convert_to_3digit_idc9(x))

# Group by ICD9 and admission 
diagnosis_icd = diagnosis_icd.groupby('hadm_id')['icd9_3digit'].unique().reset_index()
print(diagnosis_icd.head())

#Merge all 3 tables on hadm_id
df = pd.merge(patients_admissions, diagnosis_icd, how='inner', on='hadm_id')
# Sort admissions w.r.t. admission time
df = df.sort_values(by = ['subject_id', 'admittime'], ascending=True).reset_index(drop=True)

# exclude other  columns
df = df[['subject_id', 'gender', 'hadm_id', 'age', 'mortality', 'icd9_3digit']]
df = df.rename(columns={'icd9_3digit': 'icd9'})


# Statics
def mean_max_min_std(series):
    print(f"mean: {np.mean(series):.1f}, min: {np.min(series):.1f}, max: {np.max(series):.1f}, std: {np.std(series):.1f}")

print("Total # of patients:", len(df.groupby('subject_id')), '\n')
print("Total # of admissions:", len(df.groupby('hadm_id')), '\n')
print(df.groupby(['subject_id', 'gender']).size().groupby('gender').size(), '\n')

print("Age:")
mean_max_min_std(df['age'])
print()

print("# of diagnosis codes per admission:")
mean_max_min_std(df['icd9'].dropna().apply(len))
print()

print("# of admissions per patient:")
mean_max_min_std(df.groupby('subject_id')['hadm_id'].nunique())
print()

print("Mortality distribution:")
print(df.groupby(['subject_id', 'mortality']).size().groupby('mortality').size(), '\n')