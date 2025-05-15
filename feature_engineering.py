# Feature Engineering
# Convert diagnosis code to index
# To make the code machine-recognizable, we convert them from string to index.
# For example, code '008' will be converted to index 0.
# This allows us to represent diagnosis codes within an admission as a one-hot vector,
# which can be directly used by the model.

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
from torch.utils.data import Dataset

# Load the cleaned DataFrame from previous step
df = pd.read_csv("data/df_cleaned.csv")
print(df.head())

# Fix the 'icd9' column: parse string into list of codes
df['icd9'] = df['icd9'].apply(lambda x: x.strip("[]").replace("'", "").split())

# Get all unique diagnosis codes and sort them
all_codes = sorted(set([code for codes in df.icd9 for code in codes]))
print("First 10 ICD9 codes:", all_codes[:10])
print("Total unique ICD9 codes:", len(all_codes))

# Map each ICD9 code to a unique index
code2idx = {code: idx for idx, code in enumerate(all_codes)}

# Replace ICD9 code strings with corresponding index in each list
df['icd9'] = df['icd9'].apply(lambda x: [code2idx[i] for i in x])

# Join list of indices into a string (e.g., "1;5;22")
df['icd9'] = df['icd9'].apply(lambda x: ';'.join([str(i) for i in x]))

# Preview the transformed DataFrame
print(df.head())

# Train/Test split
# Get list of unique patient IDs
# We will split the data into 80% training and 20% testing sets. Normally, we should do train/validation/test splits. However, since the data is very limited, we will just do train/test splits for demonstration purpose.
all_patients = list(df.subject_id.unique().tolist())
random.shuffle(all_patients)

# Split 80% for training, 20% for testing
train_ids = all_patients[:int(len(all_patients) * 0.8)]
test_ids = all_patients[int(len(all_patients) * 0.8):]

# Filter the DataFrame by patient IDs
df_train = df[df.subject_id.isin(train_ids)].reset_index(drop=True)
df_test = df[df.subject_id.isin(test_ids)].reset_index(drop=True)

# Save splits to CSV
df_train.to_csv('df_train.csv', index=False)
df_test.to_csv('df_test.csv', index=False)

# Compute total number of diagnosis codes
Total_Num_Codes = len(all_codes)
print("Total_Num_Codes:", Total_Num_Codes)

# Helper function to read CSV and return header and rows
def read_csv(filename):
    '''Reads CSV and returns header and data rows'''
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            data.append(row)
    return header, data

# Helper function to convert indices to one-hot encoded list
def to_one_hot(label, num_classes):
    '''Converts list of indices to one-hot encoding'''
    one_hot_label = [0] * num_classes
    for i in label:
        one_hot_label[int(i)] = 1
    return one_hot_label


# Custom PyTorch Dataset for loading diagnosis and mortality data
# First, let us implement a custom dataset using PyTorch class Dataset, which will characterize the key features of the dataset we want to generate.
# We will use the diagnosis codes as input and mortality as output.
# Note: That though one patient can have multiple admissions, we will only use the diagnosis codes from the last admission since DNN cannot capture the temporal information.

class CustomDataset(Dataset):
    def __init__(self, split, total_num_codes):
        # Load CSV data
        self._df = pd.read_csv(f'{split}.csv')

        # Convert string of indices to list of integers
        self._df.icd9 = self._df.icd9.apply(lambda x: [int(i) for i in x.split(';')])

        # Build mapping from subject_id to diagnoses and label
        self._build_data_dict()

        # Store sorted subject IDs
        self._subj_ids = sorted(self._data.keys())
        self.total_num_codes = total_num_codes

    def _build_data_dict(self):
        '''Build dictionary from subject_id to diagnoses and mortality'''
        dict_data = {}
        df = self._df.groupby('subject_id').agg({
            'mortality': lambda x: x.iloc[0],
            'icd9': list
        }).reset_index()

        for _, row in df.iterrows():
            subj_id = row.subject_id
            dict_data[subj_id] = {
                'icd9': row.icd9,
                'mortality': row.mortality
            }
        self._data = dict_data

    def __len__(self):
        '''Return number of patients'''
        return len(self._subj_ids)

    def __getitem__(self, index):
        '''Return one sample: one-hot encoded diagnosis and mortality'''
        subj_id = self._subj_ids[index]
        data = self._data[subj_id]
        x = torch.tensor(to_one_hot(data['icd9'][-1], self.total_num_codes), dtype=torch.float32)
        y = torch.tensor(data['mortality'], dtype=torch.float32)
        return x, y

# Initialize datasets
train_dataset = CustomDataset('df_train', Total_Num_Codes)
test_dataset = CustomDataset('df_test', Total_Num_Codes)

print('train_dataset:', len(train_dataset))
print('test_dataset:', len(test_dataset))

# Here is an example of 𝑥, and 𝑦.
#Note: that 𝑥 is of shape 271, which means there are 271 diagnosis codes in total. It is in one-hot format. A 1 in position 𝑖 means that diagnosis code of index 𝑖 appears in the last admission.
# And 𝑦 is either 0 or 1.
x, y = train_dataset[0]
print(f'Example x: {x}')
print(f'Example y: {y}')

# Next, we will load the dataset into a dataloader so that we can we can use it to loop through the dataset for training and testing.
from torch.utils.data import DataLoader

# how many samples per batch to load
batch_size = 8
# prepare dataloaders

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print('# of train batches:', len(train_loader))
print('# of test batches:', len(test_loader))

# Check the shape of a batch from the training DataLoader
train_iter = iter(train_loader)
x, y = next(train_iter)

print('Shape of a batch x:', x.shape)  # Expected: [8, Total_Num_Codes]
print('Shape of a batch y:', y.shape)  # Expected: [8]

#You will notice that the data loader is created with a batch size of 8, and shuffle=True.
# The batch size is the number of samples we get in one iteration from the data loader and pass through our network, often called a batch and shuffle=True tells it to shuffle the dataset every time we start going through the data loader again.

