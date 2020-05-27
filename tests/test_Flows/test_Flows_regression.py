#!/usr/bin/env python3

import pytest

from prefect import Flow, Parameter, unmapped
from crawto.meta_model import MetaModel, meta_model_flow
import pandas as pd
from prefect.engine.executors import DaskExecutor
from crawto.ml_flow import data_cleaning_flow
import sqlite3
import os


def mock_db():
    if "test.db" in os.listdir():
        os.remove("test.db")


def test_data_cleaner_end_to_end_regression():
    mock_db()
    input_df = pd.read_csv("data/house-prices-advanced-regression-techniques/train.csv")
    executor = DaskExecutor()
    data_cleaner = data_cleaning_flow.run(
        input_data=input_df,
        problem="regression",
        target="SalePrice",
        features="infer",
        db_name="test.db",
        executor=executor,
    )
    assert data_cleaner.message == "All reference tasks succeeded."


def test_meta_model_regression():
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


def test_db():
    with sqlite3.connect("test.db") as conn:
        models = conn.execute("SELECT * FROM models").fetchone()
        assert len(models) > 0

        imputed_valid_df = conn.execute("SELECT * FROM imputed_valid_df").fetchone()
        assert len(imputed_valid_df) > 0

        imputed_train_df = conn.execute("SELECT * FROM imputed_train_df").fetchone()
        assert len(imputed_train_df) > 0

        transformed_train_df = conn.execute(
            "SELECT * FROM transformed_train_df"
        ).fetchone()
        assert len(transformed_train_df) > 0

        transformed_valid_df = conn.execute(
            "SELECT * FROM transformed_valid_df"
        ).fetchone()
        assert len(transformed_valid_df) > 0