import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import roc_auc_score
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset

# Load data (assuming you have the data files in the same directory)
print("Loading data...")
y_tr = pd.read_csv("../data/tox21_labels_train.csv.gz", index_col=0, compression="gzip")
y_te = pd.read_csv("../data/tox21_labels_test.csv.gz", index_col=0, compression="gzip")
x_tr_dense = pd.read_csv("../data/tox21_dense_train.csv.gz", index_col=0, compression="gzip").values
x_te_dense = pd.read_csv("../data/tox21_dense_test.csv.gz", index_col=0, compression="gzip").values
x_tr_sparse = io.mmread("../data/tox21_sparse_train.mtx.gz").tocsc()
x_te_sparse = io.mmread("../data/tox21_sparse_test.mtx.gz").tocsc()
