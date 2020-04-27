#!/usr/bin/env python3
from sklearn.metrics import r2_score


class BaselineModel(object):
    def get_params(self):
        return None


class BaselineClassificationPrediction(BaselineModel):
    def fit(
        self, X, y,
    ):
        self.y_pred = y.mode()
        return self

    def predict(
        self, X,
    ):
        return self.y_pred


class BaselineRegressionPrediction(BaselineModel):
    def fit(
        self, X, y,
    ):
        self.y_pred = y.median()
        return self

    def predict(
        self, X,
    ):
        return self.y_pred

    def score(X, y_pred):
        return r2_score(X, y_pred)
