#!/usr/bin/env python3
from sklearn.metrics import r2_score
import numpy as np


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
    def fit(self, X, y):
        self._y_pred = y.median()
        return self

    # def predict(self, X):
    #     return np.ones_like(X.shape[0]) * self._y_pred

    def score(self, X, y):
        y_true = y
        y_pred = np.ones_like(y_true) * self._y_pred
        return r2_score(y_true, y_pred)
