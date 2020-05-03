#!/usr/bin/env python3

import pytest

from prefect import Flow, Parameter, unmapped
from crawto.meta_model import MetaModel, meta_model_flow
import pandas as pd
from prefect.engine.executors import DaskExecutor
from crawto.ml_flow import data_cleaning_flow
import sqlite3


def mock_db():
    with sqlite3.connect("test.db") as conn:
        try:
            conn.execute("""DROP TABLE models""")

        except:
            pass
        try:
            conn.execute("""DROP TABLE predictions""")
        except:
            pass


def test_data_cleaner_end_to_end_regression():
    input_df = pd.read_csv("data/house-prices-advanced-regression-techniques/train.csv")
    executor = DaskExecutor()
    data_cleaner = data_cleaning_flow.run(
        input_data=input_df,
        problem="regression",
        target="SalePrice",
        features="infer",
        #    db_name="test.db",
        executor=executor,
    )
    assert data_cleaner.message == "All reference tasks succeeded."


def test_meta_model_regression():
    mock_db()
    meta = MetaModel(problem="regression", db="test.db", use_default_models=True)
    models = meta.models
    executor = DaskExecutor()
    meta_model_run = meta_model_flow.run(
        train_data="transformed_train.df",
        train_target="train_target.df",
        valid_data="transformed_valid.df",
        valid_target="valid_target.df",
        db="test.db",
        executor=executor,
    )
    assert meta_model_run.message == "All reference tasks succeeded."
