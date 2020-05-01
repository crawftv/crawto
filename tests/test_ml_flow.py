from crawto.ml_flow import fit_target_transformer, target_encoder_transform
import pytest
from hypothesis import given, settings, Verbosity, assume
from hypothesis.extra.pandas import column, data_frames, range_indexes
from hypothesis.strategies import one_of, just
from prefect import Flow


@given(
    problem=one_of(just("regression"), just("binary classification")),
    imputed_categorical_df=data_frames(
        columns=[
            column("target", dtype=int),
            column("A", dtype=int),
            column("B", dtype=int),
        ],
        index=range_indexes(min_size=1),
    ),
)
@settings(max_examples=10, verbosity=Verbosity.verbose)
def test_target_transformer(problem, imputed_categorical_df):
    # test that target transformer doesn't add nulls/0 to the df. this was a problem.
    target = "target"
    train_data = imputed_categorical_df
    columns = ["A", "B"]
    with Flow("test_target_transformer") as f:
        te = fit_target_transformer(problem, target, train_data)
        df = target_encoder_transform(te, imputed_categorical_df[columns])
    f = f.run()
    assert f.result[df].result.isna().sum().sum() == 0
