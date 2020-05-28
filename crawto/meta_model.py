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
        self,
        problem,
        db,
        model_path=None,
        use_default_models=True,
        use_dummy_models=True,
        models=None,
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
                conn.execute(
                    """CREATE TABLE fit_models (identifier text , pickled_model blob, dataset text)"""
                )
        except sqlite3.OperationalError:
            pass

        if use_default_models:
            self.add_default_models()
        if use_dummy_models:
            self.add_dummy_models()

    def add_model(self, model):
        m = Model(problem=self.problem, model=model, db=self.db)
        self.models.append(m.identifier)

    def add_dummy_models(self):
        if self.problem == "regression":
            self.add_model(DummyRegressor(strategy="median"))
            self.add_model(DummyRegressor(strategy="mean"))
        elif self.problem == "classification":
            self.add_model(DummyClassifier(strategy="most_frequent"))
            self.add_model(DummyClassifier(strategy="uniform"))
            self.add_model(DummyClassifier(strategy="stratified"))

    def add_default_models(self):
        if self.problem == "regression":
            self.add_model(DecisionTreeRegressor())
            self.add_model(Ridge())
            self.add_model(GradientBoostingRegressor())
            self.add_model(RandomForestRegressor())
            self.add_model(ElasticNet())
            self.add_model(LinearRegression())
        elif self.problem == "classification":
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
def create_predictions_table(db):
    with sqlite3.connect(db) as conn:
        query = """CREATE TABLE predictions (identifier text, scores blob, dataset text, score real) """
        conn.execute(query)


@task
def fit_model(db, model_identifier, dataset, target):
    # Model
    query = f"""SELECT * FROM models WHERE identifier = '{model_identifier}'"""
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(query).fetchone()
    model = row["pickled_model"]
    model = cloudpickle.loads(model)
    # data
    train_data_query = f"SELECT * FROM {dataset}"
    train_data = pd.read_sql(train_data_query, con=sqlite3.connect(db))
    target_data_query = f"SELECT * FROM {target}"
    target = pd.read_sql(target_data_query, con=sqlite3.connect(db))
    # fit
    fit_model = model.fit(X=train_data, y=target)
    fit_model = cloudpickle.dumps(model)
    # insert
    new_row = (row["identifier"], fit_model, dataset)
    query = "INSERT INTO fit_models VALUES (?,?,?)"
    with sqlite3.connect(db) as conn:
        conn.execute(query, new_row)


@task
def predict_model(db, model_identifier, dataset, target):
    # model
    select_models_query = (
        """SELECT pickled_model, identifier FROM fit_models WHERE identifier = (?)"""
    )
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        model, identifier = conn.execute(
            select_models_query, (model_identifier,)
        ).fetchone()
    model = row["pickled_model"]
    model = cloudpickle.loads(model)
    # data
    valid_data_query = f"SELECT * FROM {dataset}"
    valid_data = pd.read_sql(valid_data_query, con=sqlite3.connect(db))
    target_data_query = f"SELECT * FROM {target}"
    target = pd.read_sql(target_data_query, con=sqlite3.connect(db))
    # predict
    predictions = model.predict(X=valid_data)
    pickled_predictions = cloudpickle.dumps([float(i) for i in predictions])
    score = model.score(X=valid_data, y=target)
    # insert
    new_row = (row["model_identifier"], pickled_predictions, dataset, score)
    with sqlite3.connect(db) as conn:
        insert_predictions_query = "INSERT INTO predictions VALUES (?,?,?,?)"
        conn.execute(
            insert_query, (model_identifier, pickled_predictions, dataset, score)
        )


@task
def get_models(db, table_name):
    query = f"SELECT identifier FROM {table_name}"
    with sqlite3.connect(db) as conn:
        models = conn.execute(query).fetchall()
    models = [i[0] for i in models]
    return models


with Flow("meta_model_flow") as meta_model_flow:
    train_data = Parameter("train_data")
    train_target = Parameter("train_target")
    valid_data = Parameter("valid_data")
    valid_target = Parameter("valid_target")
    db = Parameter("db")
    models = get_models(db, "models")

    fit_models = fit_model.map(
        model_identifier=models,
        db=unmapped(db),
        dataset=unmapped(train_data),
        target=unmapped(train_target),
    )
    fit_models = get_models(db, "fit_models")
    predict_models = predict_model.map(
        model_identifier=fit_models,
        db=unmapped(db),
        dataset=unmapped(valid_data),
        target=unmapped(valid_target),
    )

if __name__ == "__main__":
    pass
