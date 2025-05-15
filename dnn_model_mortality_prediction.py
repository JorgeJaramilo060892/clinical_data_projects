# dnn_model_prediction/train_dnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from feature_engineering import CustomDataset, to_one_hot, Total_Num_Codes
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# DNN model
class Net(nn.Module):
    def __init__(self, input_size=Total_Num_Codes, hidden_size=16, dropout_prob=0.5):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Metrics

def classification_metrics(Y_score, Y_pred, Y_true):
    acc = accuracy_score(Y_true, Y_pred)
    auc = roc_auc_score(Y_true, Y_score)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1score = f1_score(Y_true, Y_pred)
    return acc, auc, precision, recall, f1score

def evaluate(model, loader):
    model.eval()
    all_y_true = torch.LongTensor()
    all_y_pred = torch.LongTensor()
    all_y_score = torch.FloatTensor()
    for x, y in loader:
        y_hat = model(x).view(-1)
        y_pred = (y_hat >= 0.5).int()
        all_y_true = torch.cat((all_y_true, y.to('cpu')))
        all_y_pred = torch.cat((all_y_pred, y_pred.to('cpu')))
        all_y_score = torch.cat((all_y_score, y_hat.detach().to('cpu')))
    return classification_metrics(
        all_y_score.numpy(), all_y_pred.numpy(), all_y_true.numpy())

# Load datasets
train_dataset = CustomDataset('df_train', Total_Num_Codes)
test_dataset = CustomDataset('df_test', Total_Num_Codes)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Initialize model
model = Net()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 60
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        y_hat = model(x).view(-1)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    acc, auc, precision, recall, f1 = evaluate(model, test_loader)
    print(f"ACC: {acc:.3f}, AUC: {auc:.3f}, P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")

torch.save(model.state_dict(), "dnn_model.pt")
