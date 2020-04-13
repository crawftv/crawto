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
import prefect
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
    df = pd.read_feather(filename)
    return df


@task
def load_tinydb(filename):
    tinydb = TinyDB(filename)
    return tinydb


class MetaModel(object):
    def __init__(self, problem, db, use_default_models=True, models=None):
        self.problem = problem
        self.db = db
        if models is None:
            self.models = []
        else:
            self.models = models
        self._models = []
        if use_default_models == True:
            self.add_default_models()

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, model):
        self.add_model(model)

    def add_model(self, model):
        m = Model(self.db, self.problem, model)
        self.models = self._models.append(m,)

    def add_default_models(self):
        if self.problem == "regression":
            self.add_model(modelElasticNet())
        #     self.add_model(LinearRegression())
        #     self.add_model(BaselineRegressionPrediction())
        #     self.add_model(DecisionTreeRegressor())
        #     self.add_model(Ridge())
        #     self.add_model(GradientBoostingRegressor())
        #     self.add_model(RandomForestRegressor())
        # elif self.problem == "classificiation":
        #     self.add_model(BaselineClassificationPrediction())
        #     self.add_model(DecisionTreeClassifier())
        #     self.add_model(LinearSVC())
        #     self.add_model(RandomForestClassifier())
        #     self.add_model(GradientBoostingClassifier())
        #     self.add_model(LogisticRegression())
        #     self.add_model(RidgeClassifier())
        #     self.add_model(GaussianNB())


class Model(object):
    def __init__(self, model, db, problem, name=None):
        self.problem = problem
        self.model = model
        # self.param_hash = str(
        #     hashlib.sha256(json.dumps(self.model.get_params()).encode("utf8"))
        # )
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

    def __repr__(self):
        return self.name


@task
def init_meta_model(problem, db, use_default_models=True):
    meta = MetaModel(problem, db, use_default_models=True)
    return meta


@task
def get_models(meta_model):
    logger = prefect.context.get("logger")
    logger.info(f"{meta_model.models}")


with Flow("meta_model_flow") as meta_model_flow:
    train_data = Parameter("train_data")
    valid_data = Parameter("valid_data")
    train_target = Parameter("train_target")
    problem = Parameter("problem")
    tinydb = Parameter("tinydb")
    use_default_models = Parameter("use_default_models", default=True, required=False)
    tinydb = load_tinydb(tinydb)
    transformed_train_df = load_data(train_data)
    transformed_valid_df = load_data(valid_data)
    train_target = load_data(train_target)

    meta = init_meta_model(problem, tinydb, use_default_models)
    models = get_models(meta)

    fit_models = fit_model.map(
        model=models,
        train_data=unmapped(train_data),
        target=unmapped(train_target),
        problem=unmapped(problem),
    )
    predict_models = predict_model.map(
        model=fit_models, valid_data=unmapped(valid_data),
    )

if __name__ == "__main__":
    pass
