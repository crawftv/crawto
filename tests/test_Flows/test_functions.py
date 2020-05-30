#!/usr/bin/env python3

from crawto.data_cleaning_flow import (
    df_to_sql_schema,
    create_sql_data_tables,
    save_features,
)
import pandas as pd
from prefect import Parameter, Flow


def test_df_to_sql_schema():
    d = {"col1": [1, 2], "col2": [3, 4]}
    df = pd.DataFrame(data=d)
    assert df_to_sql_schema("test", df)


def test_df_to_sql_schema1():
    d = {"col1": [1, 2], "col2": [3, 4]}
    df = pd.DataFrame(data=d)
    assert df_to_sql_schema("test", df["col1"])


def test_create_sql_data_tables():
    with Flow("test_data_tables") as test:
        db = Parameter("db")
        create_sql_data_tables(db)

    result = test.run(db=":memory:")
    assert result.message == "All reference tasks succeeded."


def test_save_features():
    with Flow("test_save_features") as test:
        db = Parameter("db_name")
        nan_features = []
        problematic_features = []
        numeric_features = []
        categorical_features = []
        imputed_train_numeric_df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
        yeo_johnson_train_transformed = pd.DataFrame(
            data={"col1": [1, 2], "col2": [3, 4]}
        )
        target_encoded_train_df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
        imputed_train_categorical_df = pd.DataFrame(
            data={"col1": [1, 2], "col2": [3, 4]}
        )
        save_features(
            db,
            nan_features,
            problematic_features,
            numeric_features,
            categorical_features,
            imputed_train_numeric_df,
            yeo_johnson_train_transformed,
            target_encoded_train_df,
            imputed_train_categorical_df,
        )

    result = test.run(db_name=":memory:")
    assert result.message == "All reference tasks succeeded."
