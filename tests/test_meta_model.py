#!/usr/bin/env python3

import pytest

from prefect.engine.executors import DaskExecutor
from crawto.meta_model import MetaModel, meta_model_flow
from tinydb import TinyDB


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
        problem="binary classification",
        models=models,
        tinydb="db.json",
        executor=executor,
    )
    assert meta_model_run.message == "All reference tasks succeeded."
