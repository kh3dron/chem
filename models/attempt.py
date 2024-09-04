import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load data (assuming you have the data files in the same directory)
y_tr = pd.read_csv("../data/tox21_labels_train.csv.gz", index_col=0, compression="gzip")
y_te = pd.read_csv("../data/tox21_labels_test.csv.gz", index_col=0, compression="gzip")
x_tr_dense = pd.read_csv("../data/tox21_dense_train.csv.gz", index_col=0, compression="gzip").values
x_te_dense = pd.read_csv("../data/tox21_dense_test.csv.gz", index_col=0, compression="gzip").values
x_tr_sparse = io.mmread("../data/tox21_sparse_train.mtx.gz").tocsc()
x_te_sparse = io.mmread("../data/tox21_sparse_test.mtx.gz").tocsc()

# Filter out very sparse features
sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].toarray()])
x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].toarray()])

# Convert to PyTorch tensors
x_tr_tensor = torch.FloatTensor(x_tr)
x_te_tensor = torch.FloatTensor(x_te)

# Define the neural network
class Tox21Net(nn.Module):
    def __init__(self, input_size):
        super(Tox21Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Remove sigmoid activation here
        return x

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())  # Apply sigmoid here
            all_labels.extend(labels.numpy())
    return roc_auc_score(all_labels, all_preds)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytorch_results = {}
for target in y_tr.columns:
    print(f"Training model for {target}...")
    
    rows_tr = np.isfinite(y_tr[target]).values
    rows_te = np.isfinite(y_te[target]).values
    
    x_train = x_tr_tensor[rows_tr]
    y_train = torch.FloatTensor(y_tr[target][rows_tr].values)
    x_test = x_te_tensor[rows_te]
    y_test = torch.FloatTensor(y_te[target][rows_te].values)
    
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    model = Tox21Net(x_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, criterion, optimizer, device, epochs=20)  # Increased epochs
    
    auc_te = evaluate_model(model, test_loader, device)
    pytorch_results[str(target)] = auc_te
    print(f"AUC for {target}: {auc_te:.4f}")

# Compare results
def comp(model_dict):
    print("Average Diff: %3.5f" % ((sum(model_dict.values()) - sum(SOTA.values())) / 12))
    print("Results greater than SOTA: %d\n" % sum(model_dict[k] > SOTA[k] for k in SOTA))
    for target in model_dict:
        print(
            "%15s: %3.5f, SOTA: %3.5f, Diff: %3.5f"
            % (target, model_dict[target], SOTA[target], model_dict[target] - SOTA[target])
        )

SOTA = {
    "NR.AhR": 0.928,
    "NR.AR": 0.828,
    "NR.AR.LBD": 0.879,
    "NR.Aromatase": 0.838,
    "NR.ER": 0.810,
    "NR.ER.LBD": 0.827,
    "NR.PPAR.gamma": 0.861,
    "SR.ARE": 0.840,
    "SR.ATAD5": 0.828,
    "SR.HSE": 0.865,
    "SR.MMP": 0.950,
    "SR.p53": 0.880,
}

print("PyTorch Neural Network Results:")
comp(pytorch_results)

# Save the PyTorch results to a file
import pickle

with open("pytorch_results.pkl", "wb") as f:
    pickle.dump(pytorch_results, f)