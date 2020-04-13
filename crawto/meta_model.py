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
from sklearn.naive_bayes import GaussianNB
from baseline_model import (
    BaselineClassificationPrediction,
    BaselineRegressionPrediction,
)
import hashlib
import json
import uuid
from tinydb import TinyDB, Query
from prefect import task, Flow, Parameter, unmapped
import pandas as pd


@task
def predict_model(model, valid_data):
    return model.predict(X=valid_data)


@task
def fit_model(model, train_data, target, problem):
    try:
        return model.fit(X=train_data, y=target)
    except AttributeError:
        logger = prefect.context.get("logger")
        logger.warning(f"Warning: Inappropriate model for {problem}.")


@task
def load_data(filename):
    return pd.read_feather(filename)


class MetaModel(object):
    def __init__(self, problem, db, use_default_models=True):
        self.problem = problem
        self.models = []
        self.db = db
        if use_default_models == True:
            self.default_models()

    def add_model_to_meta_model(self, model):
        m = Model(model, self.db, self.problem)
        self.models.append(m,)

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
            self.model(GaussianNB())


class Model(object):
    def __init__(self, model, db, problem, name=None):
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

    # def run(self):

    def __repr__(self):
        return self.name


@task
def get_models(meta_model):
    return meta_model.models


# with Flow("meta_model_flow") as meta_model_flow:
#             train_data = Parameter("train_data")
#             valid_data = Parameter("valid_data")
#             train_target = Parameter("train_target")
#             meta = Parameter("meta_model")
#             problem = Parameter("problem")
#             transformed_train_df = load_data(train_data)
#             transformed_valid_df = load_data(valid_data)
#             models = get_models(meta)
#             fit_models = fit_model.map(
#                 model=models,
#                 train_data=unmapped(train_data),
#                 target=unmapped(train_target),
#                 problem=unmapped(problem),
#             )
#             predict_models = predict_model.map(
#                 model=fit_models, valid_data=unmapped(valid_data),
#             )

if __name__ == "__main__":
    pass
