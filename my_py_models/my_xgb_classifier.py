# coding=utf-8

import xgboost as xgb


class MyXgbClassifier:
    def __init__(self, params,
                 num_boost_round=2000,
                 early_stopping_rounds=25,
                 verbose_eval=25):
        self.params = params
        self.model = None
        self.num_boost_round = num_boost_round
        self.esr = early_stopping_rounds
        self.verbose_eval = verbose_eval

    def fit(self, X_train, y_train, X_val, y_val):
        dtrain = xgb.DMatrix(X_train, y_train)
        dval = xgb.DMatrix(X_val, y_val)

        partial_model = xgb.train(self.params, dtrain, evals=[(dval, 'val')],
                                  num_boost_round=self.num_boost_round,
                                  early_stopping_rounds=self.esr,
                                  verbose_eval=self.verbose_eval)
        num_boost_round = partial_model.best_iteration

        self.model = xgb.train(dict(self.params, silent=0), dtrain,
                               num_boost_round=num_boost_round)

    def predict(self, X_test):
        if self.model is None:
            return None
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)
