import pytest
from crawto.charts import tsne, make_html
from hypothesis import given
from hypothesis.strategies import lists


@given(list())
@example()
def test_make_html():
    assert type(tsne( )) ==


def test_make_html():
    assert make_html( ) == html


