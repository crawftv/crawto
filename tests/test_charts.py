import pytest
from crawto.Charts.charts import tsne_plot, make_html
from crawto.Charts.chart_type import Data,DataSet,DataPoint
import IPython
from IPython.display import display, HTML
import json
from hypothesis import given
from hypothesis.strategies import text,builds,lists,floats


@given(builds(Data,lists(builds(DataSet,text(),lists(builds(DataPoint,floats(),floats())),text()))))
def test_tsne_chart(d):
        assert type(make_html(d)) is str





#    assert datasets.keys() ==['x','y']
#    assert datasets.

# def test_make_html():
#    label_list = ["a", "b", "c"]
#    x_list = [-1, 0, 1]
#    y_list = [1, 1, 1]
#
#
#
#    assert type(tsne(label_list, x_list, y_list)) is IPython.core.display.HTML


# def test_make_html():
#
#    assert make_html( ) == html
