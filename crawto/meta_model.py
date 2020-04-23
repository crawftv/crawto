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
from .baseline_model import (
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
import joblib
import os
from pathlib import Path


@task
def predict_model(model, valid_data):
    return model.predict(X=valid_data)


@task
def fit_model(model_path, train_data, target, problem):
    model = joblib.load(model_path)
    model.fit(X=train_data, y=target)
    joblib.dump(model, model_path)


@task
def load_data(filename):
    df = pd.read_feather(path=filename)
    return df


@task
def load_tinydb(filename):
    tinydb = TinyDB(filename)
    return tinydb


class MetaModel(object):
    def __init__(
        self, problem, db, model_path=None, use_default_models=None, models=None
    ):
        # import pdb

        # pdb.set_trace()
        self.problem = problem
        self.db = db

        if models is None:
            models = []
        self.models = models

        if model_path is None:
            model_path = "models/"
        self.model_path = model_path

        if os.path.exists(self.model_path):
            pass
        else:
            Path(self.model_path).mkdir(exist_ok=True)

        if use_default_models == None:
            use_default_models = True

        if use_default_models:
            self.add_default_models()

    def add_model(self, model):
        m = Model(
            db=self.db, problem=self.problem, model_path=self.model_path, model=model
        )
        self.models.append(m.model_path)

    def add_default_models(self):
        if self.problem == "regression":
            self.add_model(ElasticNet())

            self.add_model(LinearRegression())
            self.add_model(BaselineRegressionPrediction())
            self.add_model(DecisionTreeRegressor())
            self.add_model(Ridge())
            self.add_model(GradientBoostingRegressor())
            self.add_model(RandomForestRegressor())
        elif self.problem == "classificiation":
            self.add_model(BaselineClassificationPrediction())
            self.add_model(DecisionTreeClassifier())
            self.add_model(LinearSVC())
            self.add_model(RandomForestClassifier())
            self.add_model(GradientBoostingClassifier())
            self.add_model(LogisticRegression())
            self.add_model(RidgeClassifier())
            self.add_model(GaussianNB())


class Model(object):
    def __init__(self, name=None, model=None, db=None, problem=None, model_path="/"):
        self.problem = problem
        self.uid = uuid.uuid4()
        self.db = db
        self.model = model

        if name is None:
            self.name = f"{str(model.__class__)}-{self.uid}".replace(
                "<class '", ""
            ).replace("'>", "")
        else:
            self.name = f"{name}-{self.uid}"
        self.model_path = model_path + self.name
        # import pdb

        # pdb.set_trace()

        joblib.dump(self.model, self.model_path)

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
    return meta_model.models


with Flow("meta_model_flow") as meta_model_flow:
    train_data = Parameter("train_data")
    valid_data = Parameter("valid_data")
    train_target = Parameter("train_target")
    problem = Parameter("problem")
    models = Parameter("models")
    tinydb = Parameter("tinydb")

    use_default_models = Parameter("use_default_models", default=True, required=False)
    tinydb = load_tinydb(tinydb)
    transformed_train_df = load_data(train_data)
    transformed_valid_df = load_data(valid_data)
    train_target = load_data(train_target)

    # meta = init_meta_model(problem, tinydb, use_default_models)
    # meta = MetaModel(problem, tinydb, use_default_models=True)
    # model_path = get_models(meta)

    fit_models = fit_model.map(
        model_path=models,
        train_data=unmapped(transformed_train_df),
        target=unmapped(train_target),
        problem=unmapped(problem),
    )
    # predict_models = predict_model.map(
    #     model=fit_models, valid_data=unmapped(valid_data),
    # )

if __name__ == "__main__":
    pass
