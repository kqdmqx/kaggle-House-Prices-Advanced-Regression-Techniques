# coding=utf-8

import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from my_py_models.stacking2 import Stacking
from my_py_models.my_xgb_classifier2 import MyXgbClassifier2
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from my_py_models.utils import factorize_obj, r_square
from os.path import join

# Sample
train = pd.read_csv(join(INPUT_PATH, 'train.csv'))
test = pd.read_csv(join(INPUT_PATH, 'test.csv'))

train_ID = train.loc[train.y < 120].ID
out_ID = train.loc[train.y >= 120].ID
test_ID = test.ID

# Y
y_train = train.loc[train.y < 120].y
y_out = train.loc[train.y >= 120].y
y_all = np.append(y_train.values, y_out.values)

# Features
train_test = pd.concat([train.loc[train.y < 120].copy(),
                        train.loc[train.y >= 120].copy(),
                        test])
train_test.drop(["ID", "y"], axis=1, inplace=True)
train_test_p = factorize_obj(train_test)

X_all = train_test_p.values
print(X_all.shape)

# X_train & X_test
num_train = train_ID.shape[0]
num_out = out_ID.shape[0]
X_train = X_all[:num_train]
X_test = X_all[num_train:]

# 5-cv Xgb
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


# Save test
submission = pd.DataFrame({'ID': test_ID, 'y': pred_test[num_out:, 0]})
submission.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-XgbBaseline120-Test.csv'), index=False)

# Save out
out_pred = pd.DataFrame({'ID': out_ID, 'y': pred_test[:num_out, 0]})
out_pred.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-XgbBaseline120-Out.csv'), index=False)
# Save oof
oof_pred = pd.DataFrame({'ID': train_ID, 'y': pred_oof[:, 0]})
oof_pred.to_csv(join(
    OUTPUT_PATH, 'stacking/Submission-XgbBaseline120-OutOfFold.csv'), index=False)

# r^2 0.721090687938
print r_square(pred_oof[:, 0], y_train)
print r_square(np.append(pred_oof[:, 0], pred_test[:num_out, 0]), y_all)
