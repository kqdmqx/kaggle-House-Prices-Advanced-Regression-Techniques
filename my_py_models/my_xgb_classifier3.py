# coding=utf-8

import xgboost as xgb
from sklearn.model_selection import train_test_split


class MyXgbClassifier3:
    def __init__(self, params,
                 num_boost_round=2000,
                 early_stopping_rounds=25,
                 verbose_eval=25,
                 test_size=.2,
                 random_state=2017):
        self.params = params
        self.model = None
        self.num_boost_round = num_boost_round
        self.esr = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X_train, y_train, X_val, y_val):
        dtrain_all = xgb.DMatrix(X_train, y_train)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.test_size,
            random_state=self.random_state
        )
        dtrain = xgb.DMatrix(X_train, y_train)
        dval = xgb.DMatrix(X_val, y_val)

        partial_model = xgb.train(self.params, dtrain, evals=[(dval, 'val')],
                                  num_boost_round=self.num_boost_round,
                                  early_stopping_rounds=self.esr,
                                  verbose_eval=self.verbose_eval)
        num_boost_round = partial_model.best_iteration

        self.model = xgb.train(dict(self.params, silent=0), dtrain_all,
                               num_boost_round=num_boost_round)

    def predict(self, X_test):
        if self.model is None:
            return None
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)
