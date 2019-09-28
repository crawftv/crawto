from crawto.CrawtoDS import *
import pandas

df = pandas.read_csv("train.csv")

c = CrawtoDS(data=df, target="Survived", problem="classification")

def test_categorical_features():
    assert type(c.categorical_features) is list

def test_numeric_features():
    assert type(c.numeric_features) is list

def test_imputed_numeric_df():
    assert type(c.imputed_numeric_df) is pandas.core.frame.DataFrame
    assert c.imputed_numeric_df.isna().sum().sum() == 0

def test_imputed_categorical_df():
    assert type(c.imputed_categorical_df) is pandas.core.frame.DataFrame
    assert c.imputed_categorical_df.isna().sum().sum() == 0
