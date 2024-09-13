# make sure numpy, scipy, pandas, sklearn are installed, otherwise run
# pip install numpy scipy pandas scikit-learn
import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# load data
y_tr = pd.read_csv("../data/tox21_labels_train.csv.gz", index_col=0, compression="gzip")
y_te = pd.read_csv("../data/tox21_labels_test.csv.gz", index_col=0, compression="gzip")
x_tr_dense = pd.read_csv("../data/tox21_dense_train.csv.gz", index_col=0, compression="gzip").values
x_te_dense = pd.read_csv("../data/tox21_dense_test.csv.gz", index_col=0, compression="gzip").values
x_tr_sparse = io.mmread("../data/tox21_sparse_train.mtx.gz").tocsc()
x_te_sparse = io.mmread("../data/tox21_sparse_test.mtx.gz").tocsc()

# filter out very sparse features
sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].toarray()])
x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].toarray()])

# Best results found from leaderboard, seen here:
# http://bioinf.jku.at/research/DeepTox/tox21.html

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

# Build a random forest model for all twelve assays
forest = {}
print("[*] Testing SVM model...")
for target in y_tr.columns:
    
    print(f"[*]    Training on {target}")
    # Only consider rows where the target is not NaN - most drugs are not tested for most assays
    rows_tr = np.isfinite(y_tr[target]).values
    rows_te = np.isfinite(y_te[target]).values
    
    # SVM classification
    svm = SVC(kernel='linear')

    print(f"[*]    Fitting model with {x_tr[rows_tr].shape[0]} rows")
    svm.fit(x_tr[rows_tr], y_tr[target][rows_tr])
    y_te_pred = svm.predict(x_te[rows_te])

    
    auc_te = roc_auc_score(y_te[target][rows_te], y_te_pred)
    forest[target] = auc_te
    print(f"%15s: %3.5f, SOTA: %3.5f, Diff: %3.5f" % (target, auc_te, SOTA[target], auc_te - SOTA[target]))

print()
print("[*] Results greater than SOTA: %d\n" % sum(forest[k] > SOTA[k] for k in SOTA))