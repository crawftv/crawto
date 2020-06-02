#!/usr/bin/env python3

import os
import sqlite3

import pandas as pd
import pytest
from prefect import Flow, Parameter, unmapped
from prefect.engine.executors import DaskExecutor

from crawto.data_cleaning_flow import data_cleaning_flow
from crawto.meta_model import MetaModel, meta_model_flow


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
        db_name="test.db",
        executor=executor,
    )
    assert data_cleaner.message == "All reference tasks succeeded."


def test_meta_model_regression():
    meta = MetaModel(problem="regression", db="test.db", use_default_models=True)
    models = meta.models
    executor = DaskExecutor()
    meta_model_run = meta_model_flow.run(
        train_data="transformed_train_df",
        valid_data="transformed_valid_df",
        train_target="transformed_train_target_df",
        valid_target="transformed_valid_target_df",
        db="test.db",
        problem="regression",
        executor=executor,
    )
    assert meta_model_run.message == "All reference tasks succeeded."


def test_db():
    with sqlite3.connect("test.db") as conn:
        models = conn.execute("SELECT * FROM models").fetchone()
        assert len(models) > 0
        imputed_train_df = conn.execute("SELECT * FROM imputed_train_df").fetchone()
        assert len(imputed_train_df) > 0
        imputed_valid_df = conn.execute("SELECT * FROM imputed_valid_df").fetchone()
        assert len(imputed_valid_df) > 0
        transformed_train_df = conn.execute(
            "SELECT * FROM transformed_train_df"
        ).fetchone()
        assert len(transformed_train_df) > 0
        transformed_valid_df = conn.execute(
            "SELECT * FROM transformed_valid_df"
        ).fetchone()
        assert len(transformed_valid_df) > 0
        features = conn.execute("SELECT feature_list FROM features").fetchall()
        for i in features:
            assert type(features) is list
