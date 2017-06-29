# coding=utf-8

# copy https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697
# import numpy as np
import pandas as pd
import numpy as np
import sys
sys.path.append('../..')
from my_py_models.stacking2 import Stacking
from my_py_models.config import INPUT_PATH, OUTPUT_PATH
from my_py_models.utils import factorize_obj, get_script_title, drop_duplicate_columns
from os.path import join
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import check_array
from sklearn.pipeline import make_pipeline
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

# X_train & X_test
X_all = train_test_p.values
print(X_all.shape)
num_train = train_ID.shape[0]
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


class StackingEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack(
            (np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


clf = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber",
                                                          max_depth=3, max_features=0.55,
                                                          min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()
)
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
