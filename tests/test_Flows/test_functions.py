#!/usr/bin/env python3

from crawto.ml_flow import df_to_sql_schema,create_sql_data_tables
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
        db=Parameter("db")
        create_sql_data_tables(db)

    result = test.run(
        db = ":memory:"
    )
    assert result.message == "All reference tasks succeeded."
