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
import json
import uuid
from prefect import task, Flow, Parameter, unmapped
from prefect.tasks.database.sqlite import SQLiteQuery
import prefect
import cloudpickle
import pandas as pd
import sqlite3


class MetaModel(object):
    def __init__(
        self, problem, db, model_path=None, use_default_models=None, models=None
    ):
        self.problem = problem
        self.db = db

        if models is None:
            models = []
        self.models = models

        if use_default_models == None:
            use_default_models = True

        if use_default_models:
            self.add_default_models()

    def add_model(self, model):
        m = Model(problem=self.problem, model=model, db=self.db)
        self.models.append(m.identifier)

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
            self.add_model(LogisticRegression())
            self.add_model(LinearSVC())
            self.add_model(RandomForestClassifier())
            self.add_model(GradientBoostingClassifier())
            self.add_model(RidgeClassifier())
            self.add_model(GaussianNB())


class Model(object):
    def __init__(self, db, name=None, model=None, problem=None):
        self.problem = problem
        self.db = db
        self.model = model

        with sqlite3.connect(db) as conn:
<<<<<<< HEAD
            blob = cloudpickle.dumps(self.model)
=======
            pickled_model = cloudpickle.dumps(self.model)
>>>>>>> 049b883d97684328d3453bdebe8765cd2a4261c8
            model_type = (
                str(self.model.__class__).replace("<class '", "").replace("'>", "")
            )
            identifier = self.identifier = " ".join(
                self.model.__repr__().split()
            ).replace("'", '"')
            params = json.dumps(self.model.get_params()).replace("'", '"')
            conn.execute(
                f"""INSERT INTO models values (?,?,?,?)""",
<<<<<<< HEAD
                (model_type, params, identifier, blob),
=======
                (model_type, params, identifier, pickled_model),
>>>>>>> 049b883d97684328d3453bdebe8765cd2a4261c8
            )

    def predict(self, X):
        return self.model.predict(X)

    def __repr__(self):
        return self.name


@task
def init_meta_model(problem, db, use_default_models=True):
    meta = MetaModel(problem, db, use_default_models=True)
    return meta


@task
def predict_model(model, valid_data):
    return model.predict(X=valid_data)


@task
def fit_model(db, model_identifier, train_data, target):
    with sqlite3.connect(db) as conn:
<<<<<<< HEAD
        query = "SELECT blob FROM models WHERE identifier = (?)"
        model = conn.execute(query, model_identifier)
        model = cloudpickle.loads(model_path)
        model.fit(X=train_data, y=target)
        fit_model = cloudpickle.dumps(model)

        query = "INSERT INTO models (blob) VALUES (?)"
        conn.execute(query, fit_model)

=======
        query = f"""SELECT pickled_model FROM models WHERE identifier = '{model_identifier}'"""
        model = conn.execute(query).fetchone()[0]
        model = cloudpickle.loads(model)
        model.fit(X=train_data, y=target)
        fit_model = cloudpickle.dumps(model)

        query = "INSERT INTO models (pickled_model) VALUES (:fit_model)"
        conn.execute(query, (fit_model,))
>>>>>>> 049b883d97684328d3453bdebe8765cd2a4261c8



@task
<<<<<<< HEAD
def get_db(db):
    import pdb

    pdb.set_trace()
    return str(db)
=======
def get_models(db):
    with sqlite3.connect(db) as conn:
        query = "SELECT identifier FROM models"
        models = conn.execute(query).fetchall()
        models = [i[0] for i in models]
    return models

>>>>>>> 049b883d97684328d3453bdebe8765cd2a4261c8


@task
def load_data(filename):

    df = pd.read_feather(path=filename)
    return df


with Flow("meta_model_flow") as meta_model_flow:
    train_data = Parameter("train_data")
    valid_data = Parameter("valid_data")
    train_target = Parameter("train_target")
<<<<<<< HEAD
    db = Parameter("db")
    models = SQLiteQuery(db, "SELECT identifier FROM models")
=======
    db = Parameter("db_name")

>>>>>>> 049b883d97684328d3453bdebe8765cd2a4261c8
    transformed_train_df = load_data(train_data)
    models = get_models(db)
    transformed_valid_df = load_data(valid_data)
    train_target = load_data(train_target)

    fit_models = fit_model.map(
        model_identifier=models,
<<<<<<< HEAD
        db=db,
=======
        db=unmapped(db),
>>>>>>> 049b883d97684328d3453bdebe8765cd2a4261c8
        train_data=unmapped(transformed_train_df),
        target=unmapped(train_target),
    )
    # predict_models = predict_model.map(
    #     model=fit_models, valid_data=unmapped(valid_data),
    # )

if __name__ == "__main__":
    pass
