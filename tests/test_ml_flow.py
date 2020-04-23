#!/usr/bin/env python3

import pytest

from prefect import Flow, Parameter, unmapped
import pandas as pd
from prefect.engine.executors import DaskExecutor
from crawto.ml_flow import data_cleaning_flow
from tinydb import TinyDB


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
