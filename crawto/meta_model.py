#!/usr/bin/env python3
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import (
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
from baseline_model import (
    BaselineClassificationPrediction,
    BaselineRegressionPrediction,
)
import hashlib
import json
import uuid
from tinydb import TinyDB, query




class MetaModel(object):
    def __init__(self, problem, db):
        self.problem = problem
        self.models = []
        self.db = db

    def add_model_to_meta_model(self, model):
        m = Model(model, db, self.problem)
        self.models.append(m)

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
        elif self.problem == "classificiation":
            self.model(BaselineClassificationPrediction())
            self.model(DecisionTreeClassifier())
            self.model(LinearSVC())
            self.model(RandomForestClassifier())
            self.model(GradientBoostingClassifier())
            self.model(LogisticRegression())
            self.model(RidgeClassifier())


class Model(object):
<<<<<<< HEAD
    def __init__(self, model, db, problem, name=None):
=======
    def __init__(self, model, problem, name=None):
>>>>>>> 7eb5d5923b5dcbd9fc2354c4df6c41ad22fbb330
        self.problem = problem
        self.model = model
        self.param_hash = str(
            hashlib.sha256(json.dumps(self.model.get_params()).encode("utf8"))
        )
        self.uid = uuid.uuid4()
        self.db = db
        if name is None:
            self.name = f"{model.__class__}-{self.uid}"
        else:
            self.name = f"{name}-{self.uid}"

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __repr_(self):
        return self.name


if __name__ == "__main__":
    pass
