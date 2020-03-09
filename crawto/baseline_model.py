#!/usr/bin/env python3


class BaselineClassificationPrediction:

    def fit(self,X,y,):
        self.y_pred = y.mode()
        return self

    def predict(self,X,):
        return self.y_pred

class BaselineRegressionPrediction:
    def fit(self,X,y,):
        self.y_pred = y.median()
        return self
    def predict(self,X,):
        return self.y_pred
