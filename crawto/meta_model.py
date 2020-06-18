#!/usr/bin/env python3
import json
import sqlite3
from typing import Union
import cloudpickle
import pandas as pd
import prefect
from prefect import Flow, Parameter, task, unmapped
from prefect.core.edge import Edge
from prefect.engine.executors import DaskExecutor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class MetaModel:
    def __init__(
        self,
        problem: str,
        db_name: str,
        use_default_models=True,
        use_dummy_models=True,
        models=None,
    ):
        self.problem = problem
        self.db_name = db_name

        if models is None:
            models = []
        self.models = models
        try:
            with sqlite3.connect(self.db_name) as conn:
                conn.execute(
                    """CREATE TABLE models (
                    model_type text, 
                    params text, 
                    identifier text PRIMARY KEY, 
                    pickled_model blob
                    )"""
                )
                conn.execute(
                    """CREATE TABLE fit_models (
                        identifier text, 
                        pickled_model blob, 
                        dataset text
                        )"""
                )
        except sqlite3.OperationalError:
            pass

        if use_default_models:
            self.add_default_models()
        if use_dummy_models:
            self.add_dummy_models()

    def add_model(self, model) -> None:
        model = Model(problem=self.problem, model=model, db_name=self.db_name)
        self.models.append(model.identifier)

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


class Model:
    def __init__(self, db_name: str, model, problem: Union[str, None]):
        self.problem = problem
        self.db_name = db_name
        self.model = model

        with sqlite3.connect(self.db_name) as conn:
            blob = cloudpickle.dumps(self.model)
            model_type = (
                str(self.model.__class__).replace("<class '", "").replace("'>", "")
            )
            identifier = self.identifier = " ".join(
                self.model.__repr__().split()
            ).replace("'", '"')
            params = json.dumps(self.model.get_params()).replace("'", '"')
            conn.execute(
                """REPLACE INTO models values (?,?,?,?)""",
                (model_type, params, identifier, blob),
            )


@task(name="init_meta_models")
def init_meta_model(problem: str, db_name: str, use_default_models=True) -> None:
    MetaModel(problem=problem, db_name=db_name, use_default_models=True)


@task(name="create_predictions_table")
def create_predictions_table(db_name: str) -> None:
    with sqlite3.connect(db_name) as conn:
        query = """CREATE TABLE predictions 
        (identifier text, 
        predictions blob,
        predict_proba blob,
        dataset text, 
        score real) """
        conn.execute(query)


@task(name="fit_model")
def fit_model(db_name: str, model_identifier: str, dataset: str, target: str) -> None:
    # Model
    query = f"""SELECT * FROM models WHERE identifier = '{model_identifier}'"""
    with sqlite3.connect(db_name) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(query).fetchone()
    model = row["pickled_model"]
    model = cloudpickle.loads(model)
    # data
    train_data_query = f"SELECT * FROM {dataset}"
    train_data = pd.read_sql(train_data_query, con=sqlite3.connect(db_name))
    target_data_query = f"SELECT * FROM {target}"
    target = pd.read_sql(target_data_query, con=sqlite3.connect(db_name))
    # fit
    fit_model = model.fit(X=train_data, y=target)
    fit_model = cloudpickle.dumps(model)
    # insert
    new_row = (row["identifier"], fit_model, dataset)
    query = "INSERT INTO fit_models VALUES (?,?,?)"
    with sqlite3.connect(db_name) as conn:
        conn.execute(query, new_row)


@task(name="predict_model")
def predict_model(*,db_name: str, model_identifier:str, dataset:str, target:str,problem:str):
    with sqlite3.connect(db_name) as conn:
        conn.row_factory = sqlite3.Row
        # model
        select_models_query = """
        SELECT pickled_model, identifier 
        FROM fit_models 
        WHERE identifier = (?)"""
        row = conn.execute(select_models_query, (model_identifier,)).fetchone()
    model = row["pickled_model"]
    model = cloudpickle.loads(model)
    # data
    valid_data_query = f"SELECT * FROM {dataset}"
    valid_data = pd.read_sql(valid_data_query, con=sqlite3.connect(db_name))
    target_data_query = f"SELECT * FROM {target}"
    target = pd.read_sql(target_data_query, con=sqlite3.connect(db_name))
    # predict
    predictions = model.predict(X=valid_data)
    pickled_predictions = cloudpickle.dumps([float(i) for i in predictions])
    if problem == "classification":
        try:
            predict_proba = model.predict_proba(X=valid_data)
            pickled_proba = cloudpickle.dumps([float(i) for i in predict_proba.T[1]])
        except AttributeError:
            pickled_proba = None
    else:
        pickled_proba = None
    score = model.score(X=valid_data, y=target)
    # insert
    with sqlite3.connect(db_name) as conn:
        insert_predictions_query = "INSERT INTO predictions VALUES (?,?,?,?,?)"
        conn.execute(
            insert_predictions_query,
            (model_identifier, pickled_predictions,pickled_proba, dataset, score),
        )




@task(name="get_models")
def get_models(db_name: str, table_name: str):
    query = f"SELECT identifier FROM {table_name}"
    with sqlite3.connect(db_name) as conn:
        models = conn.execute(query).fetchall()
    models = [i[0] for i in models]
    logger = prefect.context.get("logger")
    logger.info(f"{table_name}: {models}")

    return models


with Flow("meta_model_flow") as meta_model_flow:
    train_data = Parameter("train_data")
    train_target = Parameter("train_target")
    valid_data = Parameter("valid_data")
    valid_target = Parameter("valid_target")
    problem = Parameter("problem")
    db_name = Parameter("db_name")

    init_meta_model = init_meta_model(problem, db_name)
    create_predictions_table = create_predictions_table(db_name)
    models = get_models(
        db_name, "models", upstream_tasks=[init_meta_model, create_predictions_table]
    )

    fit_models = fit_model.map(
        model_identifier=models,
        db_name=unmapped(db_name),
        dataset=unmapped(train_data),
        target=unmapped(train_target),
    )

    fitted_models = get_models(db_name, "fit_models", upstream_tasks=[fit_models])

    predict_models = predict_model.map(
        model_identifier=fitted_models,
        db_name=unmapped(db_name),
        dataset=unmapped(valid_data),
        target=unmapped(valid_target),
        problem = unmapped(problem)
    )


def run_meta_model(meta_model_flow, problem: str, db_name: str) -> None:
    executor = DaskExecutor()
    flow_state = meta_model_flow.run(
        train_data="transformed_train_df",
        valid_data="transformed_valid_df",
        train_target="transformed_train_target_df",
        valid_target="transformed_valid_target_df",
        problem=problem,
        db_name=db_name,
        executor=executor,
    )
    meta_model_flow.visualize(flow_state=flow_state)
