# coding=utf-8

import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from my_py_models.stacking2 import Stacking
from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from my_py_models.utils import factorize_obj, get_script_title, drop_duplicate_columns
from os.path import join
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# Sample
title = get_script_title(__file__)
print title
train = pd.read_csv(join(INPUT_PATH, 'train.csv'))
test = pd.read_csv(join(INPUT_PATH, 'test.csv'))
train_ID = train.ID
test_ID = test.ID

# Y
y_train = train.y

# Features
train_test = pd.concat([train, test])
train_test.drop(["ID", "y"], axis=1, inplace=True)
train_test_p = factorize_obj(train_test)
train_test_p = drop_duplicate_columns(train_test_p)

############################################################
# Add decomposition feature
############################################################
n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results = tsvd.fit_transform(train_test_p)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca_results = pca.fit_transform(train_test_p)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica_results = ica.fit_transform(train_test_p)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results = grp.fit_transform(train_test_p)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results = srp.fit_transform(train_test_p)

# save columns list before adding the decomposition components
usable_columns = train_test_p.columns

print train_test_p.shape
print tsvd_results.shape, type(tsvd_results)

# # Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train_test_p['pca_' + str(i)] = pca_results[:, i - 1]
    train_test_p['ica_' + str(i)] = ica_results[:, i - 1]
    train_test_p['tsvd_' + str(i)] = tsvd_results[:, i - 1]
    train_test_p['grp_' + str(i)] = grp_results[:, i - 1]
    train_test_p['srp_' + str(i)] = srp_results[:, i - 1]

##############################################################

# X_train & X_test
X_all = train_test_p.values
print(X_all.shape)
num_train = train_ID.shape[0]
X_train = X_all[:num_train]
X_test = X_all[num_train:]

# feature selection
clf = Lasso(normalize=True, alpha=0.005)
stacking = Stacking(5, [clf])
pred_oof, pred_test = stacking.fit_predict(X_train, y_train, X_test)

selected_idx = np.zeros(X_train.shape[1])
for esimator in stacking.estimators:
    coef = esimator.coef_
    selected_idx = np.logical_or(selected_idx, coef != 0)

selected_columns = list(np.array(train_test_p.columns)[selected_idx])

X_all = train_test_p[selected_columns].values
print(X_all.shape)
num_train = train_ID.shape[0]
X_train = X_all[:num_train]
X_test = X_all[num_train:]

# real work
xgb_params = {
    'n_trees': 500,
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

clf = MyXgbClassifier2(xgb_params)
stacking = Stacking(5, [clf])
pred_oof, pred_test = stacking.fit_predict(X_train, y_train, X_test)

# r^2 0.56200717888
for pred_oof_single in pred_oof.T:
    print r2_score(y_train, pred_oof_single)

# Save test
submission = pd.DataFrame({'ID': test_ID, 'y': pred_test[:, 0]})
submission.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-{}-Test.csv'.format(title)), index=False)

# Save oof
oof_pred = pd.DataFrame({'ID': train_ID, 'y': pred_oof[:, 0]})
oof_pred.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-{}-OutOfFold.csv'.format(title)), index=False)
