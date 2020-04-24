#!/usr/bin/env python3

import pytest

from prefect import Flow, Parameter, unmapped
from crawto.meta_model import MetaModel, meta_model_flow
import pandas as pd
from prefect.engine.executors import DaskExecutor
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


def test_data_cleaner_end_to_end_regression():
    input_df = pd.read_csv("data/house-prices-advanced-regression-techniques/train.csv")
    executor = DaskExecutor()
    data_cleaner = data_cleaning_flow.run(
        input_data=input_df,
        problem="regression",
        target="SalePrice",
        features="infer",
        executor=executor,
    )
    assert data_cleaner.message == "All reference tasks succeeded."


def test_meta_model_regression():
    meta = MetaModel(problem="regression", db="test.db", use_default_models=True)
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
