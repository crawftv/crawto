import pytest
from crawto.charts import *
import IPython
from IPython.display import display, HTML

#label_list = ["a", "b", "c"]
#x_list = [-1, 0, 1]
#y_list = [1, 1, 1]


def test_make_html():
    label_list = ["a", "b", "c"]
    x_list = [-1, 0, 1]
    y_list = [1, 1, 1]
    assert type(tsne(label_list, x_list, y_list)) is IPython.core.display.HTML


#def test_make_html():
#    
#    assert make_html( ) == html
