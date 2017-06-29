# coding=utf-8

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../..')
from my_py_models.stacking2 import Stacking
# from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from my_py_models.utils import get_script_title
from os.path import join
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# stacking_dir = '../../output/stacking/'
title = get_script_title(__file__)
print title
stacking_dir = join(OUTPUT_PATH, 'stacking')
oof_format = 'Submission-{}-OutOfFold.csv'
test_format = 'Submission-{}-Test.csv'

model_list = map(get_script_title, os.listdir('../../python-script/stacking/'))
model_list.remove('XgbBaseline120')
model_list = filter(lambda x: not x.startswith('Submission-Ensemble'), model_list)
# model_list.remove('LassoFullDummies')
# model_list.remove('EnsembleXgb')
# model_list.remove('EnsembleLasso')
# model_list.remove('EnsembleLassoDecomposition')
# model_list.remove('LassoLarsDecomposition')
# model_list.remove('LassoLarsPartDummies')
index_col = 'ID'
tar_col = 'y'


def load_stacking_data(model_list,
                       index_col=index_col,
                       tar_col=tar_col,
                       path_format=oof_format,
                       stacking_dir=stacking_dir,
                       silence=False):
    oof_dict = {}
    for model_str in model_list:
        model_path = os.path.join(stacking_dir, path_format.format(model_str))
        if os.path.isfile(model_path):
            if not silence:
                print 'load file', model_path
            oof_dict[model_str] = pd.read_csv(
                model_path, index_col=index_col)[tar_col]
    return pd.DataFrame(oof_dict)


def load_stacking_predictions(model_list,
                              train_path,
                              index_col=index_col,
                              tar_col=tar_col,
                              oof_format=oof_format,
                              test_format=test_format,
                              stacking_dir=stacking_dir,
                              silence=False):
    oof_df = load_stacking_data(model_list,
                                index_col=index_col,
                                tar_col=tar_col,
                                path_format=oof_format,
                                stacking_dir=stacking_dir,
                                silence=silence)
    oof_df[tar_col] = pd.read_csv(train_path, index_col=index_col)[tar_col]
    test_df = load_stacking_data(model_list,
                                 index_col=index_col,
                                 tar_col=tar_col,
                                 path_format=test_format,
                                 stacking_dir=stacking_dir,
                                 silence=silence)
    return oof_df, test_df


oof_df, test_df = load_stacking_predictions(
    model_list, join(INPUT_PATH, 'train.csv'))

# oof cv
for col in oof_df.columns:
    print col, r2_score(oof_df.y, oof_df[col])


train = oof_df
test = test_df
train_test_p = pd.concat([train, test])
train_test_p.drop(["y"], axis=1, inplace=True)

############################################################
# Add decomposition feature
############################################################
n_comp = 5

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


# data transform
y_train = oof_df.y
train_ID = oof_df.index.values
test_ID = test_df.index.values
X_all = train_test_p.values
print(X_all.shape)
num_train = train_ID.shape[0]
X_train = X_all[:num_train]
X_test = X_all[num_train:]

# 5cv search
# for alpha in np.linspace(.25, .31, 10):
#     clf = Lasso(normalize=False, alpha=alpha)
#     stacking = Stacking(5, [clf])
#     pred_oof, pred_test = stacking.fit_predict(X_train, y_train, X_test)

#     # r^2 0.56200717888
#     for pred_oof_single in pred_oof.T:
#         print round(alpha, 2), r2_score(y_train, pred_oof_single)

alpha = 0.27
clf = Lasso(normalize=False, alpha=alpha)
stacking = Stacking(5, [clf])
pred_oof, pred_test = stacking.fit_predict(X_train, y_train, X_test)

# r^2 0.568597016832
for pred_oof_single in pred_oof.T:
    print round(alpha, 2), r2_score(y_train, pred_oof_single)


# Save test
submission = pd.DataFrame({'ID': test_ID, 'y': pred_test[:, 0]})
submission.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-{}-Test.csv'.format(title)), index=False)

# Save oof
oof_pred = pd.DataFrame({'ID': train_ID, 'y': pred_oof[:, 0]})
oof_pred.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-{}-OutOfFold.csv'.format(title)), index=False)

# 0.27 0.571376126373
