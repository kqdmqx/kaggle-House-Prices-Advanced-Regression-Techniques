# coding=utf-8

# import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from my_py_models.stacking import Stacking
from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from my_py_models.utils import factorize_obj, get_script_title, drop_duplicate_columns
from os.path import join
from sklearn.metrics import r2_score
# import seaborn as sns
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

# Features
train_test = pd.concat([train, test])
train_test_p = factorize_obj(train_test)
train_test_p = drop_duplicate_columns(train_test_p)
train_test_p.drop(["X0", "ID", "y"], axis=1, inplace=True)

X_all = train_test_p.values
print(X_all.shape)

# X_train & X_test
num_train = train_ID.shape[0]
X_train = X_all[:num_train]
X_test = X_all[num_train:]

# 5-cv Xgb
xgb_params = {
    'n_trees': 500,
    'eta': 0.05,
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

plt.scatter(y_train, pred_oof, alpha=.1)
plt.plot((75, 120), (75, 120), c='r')
plt.show()
