#!/usr/bin/env python3

import pytest
from prefect.engine.executors import DaskExecutor
from crawto.meta_model import MetaModel, meta_model_flow
from tinydb import TinyDB
from prefect import Flow, Parameter, unmapped
import pandas as pd
from crawto.ml_flow import data_cleaning_flow
import sqlite3
with sqlite3.connect("test.db") as conn:
    try:
        conn.execute("""DROP TABLE models""")
    except:
        pass
    conn.execute(
        """CREATE TABLE models (model_type text, params text, identifier text PRIMARY KEY, pickled_model blob)"""
    )

def test_data_cleaner_end_to_end_classification():
    input_df = pd.read_csv("data/titanic/train.csv")
    test = pd.read_csv("data/titanic/test.csv")
    executor = DaskExecutor()
    data_cleaner = data_cleaning_flow.run(
        input_data=input_df,
        problem="binary classification",
        target="Survived",
        features="infer",
        executor=executor,
    )
    assert data_cleaner.message == "All reference tasks succeeded."


def test_meta_model_classification():
    meta = MetaModel(
        "binary classification", TinyDB("db.json"), use_default_models=True
    )
    models = meta.models
    executor = DaskExecutor()
    meta_model_run = meta_model_flow.run(
        train_data="transformed_train.df",
        valid_data="transformed_valid.df",
        train_target="train_target.df",
        db_name="test.db",
        executor=executor,
    )
    assert meta_model_run.message == "All reference tasks succeeded."
