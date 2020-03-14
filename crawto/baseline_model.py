#!/usr/bin/env python3
class BaselineModel(object):

    def get_params(self):
        return None

class BaselineClassificationPrediction(BaselineModel):

    def fit(self,X,y,):
        self.y_pred = y.mode()
        return self

    def predict(self,X,):
        return self.y_pred

class BaselineRegressionPrediction(BaselineModel):
    def fit(self,X,y,):
        self.y_pred = y.median()
        return self
    def predict(self,X,):
        return self.y_pred
