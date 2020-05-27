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
from sklearn.dummy import DummyClassifier, DummyRegressor
import json
import prefect
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
        try:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """CREATE TABLE models (model_type text, params text, identifier text PRIMARY KEY, pickled_model blob)"""
                )
        except sqlite3.OperationalError:
            pass

        if use_default_models == None:
            use_default_models = True

        if use_default_models:
            self.add_default_models()

    def add_model(self, model):
        m = Model(problem=self.problem, model=model, db=self.db)
        self.models.append(m.identifier)

    def add_default_models(self):
        if self.problem == "regression":
            self.add_model(DummyRegressor(strategy="median"))
            self.add_model(DummyRegressor(strategy="mean"))
            self.add_model(DecisionTreeRegressor())
            self.add_model(Ridge())
            self.add_model(GradientBoostingRegressor())
            self.add_model(RandomForestRegressor())
            self.add_model(ElasticNet())
            self.add_model(LinearRegression())
        elif self.problem == "classification":
            self.add_model(DummyClassifier(strategy="most_frequent"))
            self.add_model(DummyClassifier(strategy="uniform"))
            self.add_model(DummyClassifier(strategy="stratified"))
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
            blob = cloudpickle.dumps(self.model)
            model_type = (
                str(self.model.__class__).replace("<class '", "").replace("'>", "")
            )
            identifier = self.identifier = " ".join(
                self.model.__repr__().split()
            ).replace("'", '"')
            params = json.dumps(self.model.get_params()).replace("'", '"')
            conn.execute(
                f"""REPLACE INTO models values (?,?,?,?)""",
                (model_type, params, identifier, blob),
            )

    def __repr__(self):
        return self.name


@task
def init_meta_model(problem, db, use_default_models=True):
    meta = MetaModel(problem, db, use_default_models=True)
    return meta


@task
def fit_model(db, model_identifier, train_data, target):
    with sqlite3.connect(db) as conn:
        query = f"""SELECT pickled_model FROM models WHERE identifier = '{model_identifier}'"""
        model = conn.execute(query).fetchone()[0]
        model = cloudpickle.loads(model)
        model = model.fit(X=train_data, y=target)
        fit_model = cloudpickle.dumps(model)

        query = "UPDATE models SET pickled_model = (?) WHERE identifier = (?)" ""
        conn.execute(query, (fit_model, model_identifier))


@task
def create_predictions_table(db):
    with sqlite3.connect(db) as conn:
        query = """CREATE TABLE predictions (identifier text, scores blob, dataset text, score real) """
        conn.execute(query)


@task
def predict_model(db, model_identifier, valid_data, target, fit_model):
    with sqlite3.connect(db) as conn:
        select_query = (
            """SELECT pickled_model, identifier FROM models WHERE identifier = (?)"""
        )
        model, identifier = conn.execute(select_query, (model_identifier,)).fetchone()
        model = cloudpickle.loads(model)
        predictions = model.predict(X=valid_data)
        pickled_predictions = cloudpickle.dumps([float(i) for i in predictions])
        score = model.score(X=valid_data, y=target)
        insert_query = "INSERT INTO predictions VALUES (?,?,?,?)"
#        conn.execute(insert_query, (model_identifier, pickled_predictions, dataset, score))


@task
def get_models(db):
    with sqlite3.connect(db) as conn:
        query = "SELECT identifier FROM models"
        models = conn.execute(query).fetchall()
        models = [i[0] for i in models]
    return models



@task
def load_data(filename):
    df = pd.read_csv(filename)
    return df


with Flow("meta_model_flow") as meta_model_flow:
    train_data = Parameter("train_data")
    train_target = Parameter("train_target")
    valid_data = Parameter("valid_data")
    valid_target = Parameter("valid_target")
    db = Parameter("db")
    transformed_train_df = load_data(train_data)
    models = get_models(db)
    transformed_valid_df = load_data(valid_data)
    train_target = load_data(train_target)
    valid_target = load_data(valid_target)

    fit_models = fit_model.map(
        model_identifier=models,
        db=unmapped(db),
        train_data=unmapped(transformed_train_df),
        target=unmapped(train_target),
    )
    predict_models = predict_model.map(
        model_identifier=models,
        db=unmapped(db),
        valid_data=unmapped(transformed_valid_df),
        target=unmapped(valid_target),
        fit_model=fit_models,
    )

if __name__ == "__main__":
    pass
