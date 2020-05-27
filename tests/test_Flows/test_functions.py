#!/usr/bin/env python3

from crawto.ml_flow import df_to_sql_schema
import pandas as pd

def test_df_to_sql_schema():
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    assert df_to_sql_schema("test",df)

def test_df_to_sql_schema1():
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    assert df_to_sql_schema("test", df["col1"])
