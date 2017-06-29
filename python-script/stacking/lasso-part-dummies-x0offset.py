# coding=utf-8

# import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from my_py_models.stacking2 import Stacking
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from my_py_models.utils import get_script_title, drop_duplicate_columns
from os.path import join
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Sample
title = get_script_title(__file__)
print title
train = pd.read_csv(join(INPUT_PATH, 'train.csv'))
test = pd.read_csv(join(INPUT_PATH, 'test.csv'))
train_ID = train.ID
test_ID = test.ID

# Y
y_train = train.y
X0_mean_dict = train.groupby('X0').y.agg('mean').to_dict()
# X0_std_dict = train.groupby('X0').y.agg('std').to_dict()
X0_mean = train['X0'].map(X0_mean_dict)
y_train = y_train - X0_mean
sns.kdeplot(y_train)
plt.show()

plt.scatter(train.y, y_train)
plt.show()

plt.scatter(X0_mean, y_train)
plt.show()

# Features
train_test = pd.concat([train, test])
train_test.drop(["ID", "y"], axis=1, inplace=True)
train_test_numeric = train_test.select_dtypes(exclude=['object'])
train_test_obj = train_test.select_dtypes(include=['object']).copy()
train_test_dummies = pd.get_dummies(train_test_obj)
train_test_p = pd.concat([train_test_numeric, train_test_dummies], axis=1)
train_test_p = drop_duplicate_columns(train_test_p)

# X_train & X_test
X_all = train_test_p.values
print(X_all.shape)
num_train = train_ID.shape[0]
X_train = X_all[:num_train]
X_test = X_all[num_train:]

# 5cv
clf = Lasso(normalize=True, alpha=0.0025)
stacking = Stacking(5, [clf])
pred_oof, pred_test = stacking.fit_predict(X_train, y_train, X_test)

# r^2 0.56200717888
# r^2 0.56255125154
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
