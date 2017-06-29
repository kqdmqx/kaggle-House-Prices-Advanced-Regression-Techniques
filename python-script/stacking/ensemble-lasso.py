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
# from sklearn.decomposition import PCA, FastICA, TruncatedSVD
# from sklearn.random_projection import GaussianRandomProjection
# from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# stacking_dir = '../../output/stacking/'
title = get_script_title(__file__)
print title
stacking_dir = join(OUTPUT_PATH, 'stacking')
oof_format = 'Submission-{}-OutOfFold.csv'
test_format = 'Submission-{}-Test.csv'
model_list = ['Lasso', 'LassoLars', 'Decomposition',
              'RandomForestRegressor', 'XgbBaseline']

model_list = map(get_script_title, os.listdir('../../python-script/stacking/'))
model_list.remove('XgbBaseline120')
model_list = filter(lambda x: not x.startswith('Submission-Ensemble'), model_list)
# model_list.remove('XgbBaseline120')
# model_list.remove('LassoFullDummies')
# model_list.remove('EnsembleXgb')
# model_list.remove('EnsembleLasso')
# model_list.remove('EnsembleLassoDecomposition')
# model_list.remove('EnsembleBayesianRidge')
# model_list.remove('PiplineBaseline')
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

# data transform
model_list = filter(lambda x: x in oof_df.columns, model_list)
y_train = oof_df.y
X_train = oof_df[model_list].values
X_test = test_df[model_list].values
train_ID = oof_df.index.values
test_ID = test_df.index.values

# 5cv
clf = Lasso(normalize=False, alpha=0.25)
stacking = Stacking(5, [clf], metric=r2_score)
pred_oof, pred_test = stacking.fit_predict(X_train, y_train, X_test)

# r^2 0.56200717888
for pred_oof_single in pred_oof.T:
    print r2_score(y_train, pred_oof_single)
metric_result = stacking.metric_result
print np.mean(metric_result), np.std(metric_result)

# Save test
submission = pd.DataFrame({'ID': test_ID, 'y': pred_test[:, 0]})
submission.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-{}-Test.csv'.format(title)), index=False)

# Save oof
oof_pred = pd.DataFrame({'ID': train_ID, 'y': pred_oof[:, 0]})
oof_pred.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-{}-OutOfFold.csv'.format(title)), index=False)

# 0.57113730996
