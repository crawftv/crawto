#!/usr/bin/env python3

import pytest

from prefect import Flow, Parameter, unmapped
from crawto.meta_model import MetaModel, meta_model_flow
import pandas as pd
from prefect.engine.executors import DaskExecutor
from crawto.ml_flow import data_cleaning_flow
from tinydb import TinyDB


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
    meta = MetaModel("regression", TinyDB("db.json"), use_default_models=True)
    models = meta.models
    executor = DaskExecutor()
    meta_model_run = meta_model_flow.run(
        train_data="transformed_train.df",
        valid_data="transformed_valid.df",
        train_target="train_target.df",
        problem="regression",
        models=models,
        tinydb="db.json",
        executor=executor,
    )
    assert meta_model_run.message == "All reference tasks succeeded."
