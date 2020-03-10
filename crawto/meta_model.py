#!/usr/bin/env python3
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import(

    DecisionTreeRegressor,
    DecisionTreeClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    ElasticNet,
    Ridge,
    RidgeClassifier,
)
from .baseline_model import (
    BaselineClassificationPrediction,
    BaselineRegressionPrediction
    )


class MetaModel(object):
    def __init__(self, problem):
        self.problem = problem
        self.models = []

    def add_model_to_meta_model(self, model):
        self.models.append(model, self.problem)

    def model(self, model):
        self.add_model_to_meta_model(model,)
    def default_models(self):
        if self.problem == "regression":
            self.model(ElasticNet())
            self.model(LinearRegression())
            self.model(BaselineRegressionPrediction())
            self.model(DecisionTreeRegressor())
            self.model(Ridge())
            self.model(GradientBoostingRegressor())
            self.model(RandomForestRegressor())
            self.model(LogisticRegression())
        elif self.problem == "classificiation":
            self.model(BaselineClassificationPrediction())
            self.model(DecisionTreeClassifier())
            self.model(LinearSVC())
            self.model(RandomForestClassifier())
            self.model(GradientBoostingClassifier())
            self.model(RidgeClassifier())

class Model(object):
    def __init__(self, model, problem):
        self.problem = problem
        self.model = model

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
