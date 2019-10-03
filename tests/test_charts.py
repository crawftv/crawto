import pytest
from crawto.charts import *
import IPython
from IPython.display import display, HTML
import json

label_list = ["a", "b", "c"]
x_list = [-1, 0, 1]
y_list = [1, 1, 1]


def test_make_html():

    data = {
        "datasets": [{"label": "data1", "data": [{"x": 1, "y": 1}, {"x": 2, "y": 2}]}]
    }
    data = json.dumps(data)

    assert type(make_html(data)) is str


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
