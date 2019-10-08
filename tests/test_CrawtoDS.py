from crawto.CrawtoDS import *
import pandas

df = pandas.read_csv("train.csv")

c = CrawtoDS(data=df, target="Survived", problem="classification")


def test_class_initialization():
    assert isinstance(c, CrawtoDS) is True


def test_categorical_features():
    assert type(c.categorical_features) is list


def test_numeric_features():
    assert type(c.numeric_features) is list


def test_train_imputed_numeric_df():
    assert type(c.train_imputed_numeric_df) is pandas.core.frame.DataFrame
    assert c.train_imputed_numeric_df.isna().sum().sum() == 0
    # assert c.train_imputed_numeric_df.shape == (len(c.input_data),len(c.numeric_features))


def test_train_imputed_categorical_df():
    assert type(c.train_imputed_categorical_df) is pandas.core.frame.DataFrame
    assert c.train_imputed_categorical_df.isna().sum().sum() == 0
    # assert c.train_imputed_categorical_df.shape == (len(c.input_data),len(c.categorical_features))
