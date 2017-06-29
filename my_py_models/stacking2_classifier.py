# coding=utf-8

import numpy as np
from sklearn.model_selection import KFold
from copy import copy


class Stacking:
    def __init__(self, n_folds, base_models, metric=None, random_state=2016):
        '''
        Ensemble Stacking Level
        Parameters:
            param1: self
            param2: cv ford num
            param3: list of clf
        '''
        self.n_folds = n_folds
        self.base_models = base_models
        self.random_state = random_state
        self.metric = metric
        self.metric_result = []
        self.estimators = []
        self.kf = None

    def fit_predict(self, X, y, T):
        '''
        Foreach base_model: fit X, y
        Parameters:
            param1: self
            param2：X, ndarray, features of train set
            param3: y, ndarray, target values of train set
            param4: X_test, ndarray, features of test set
        Return:
            out_of_fold_pred: ndarray, X.shape[0] * len(base_models)
            test_pred: ndarray, T.shape[0] * len(base_models)

        '''
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)
        self.kf = kf
        folds = list(kf.split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, estimator_prototype in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                estimator = copy(estimator_prototype)
                self.estimators.append(estimator)
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_holdout)[:]
                if self.metric is not None:
                    self.metric_result.append(self.metric(y_holdout, y_pred))
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = estimator.predict(T)[:]
            S_test[:, i] = S_test_i.mean(1)

        out_of_fold_pred = S_train
        test_pred = S_test
        return out_of_fold_pred, test_pred

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass
