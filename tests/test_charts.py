import pytest
import IPython
from IPython.display import display, HTML
from crawto.charts.charts import Plot, ScatterChart, BarChart, LineChart, Chart_type
from hypothesis import given
from hypothesis.strategies import (
    text,
    builds,
    iterables,
    floats,
    one_of,
    integers,
    composite,
)


class TestChart:
    @composite
    def generate_good_Chart_data(draw):
        n = draw(integers(0, 50))
        c = one_of(builds(BarChart), builds(LineChart), builds(ScatterChart))
        label = text()
        x = iterables(one_of(integers(), floats()), min_size=n, max_size=n)
        y = iterables(one_of(integers(), floats()), min_size=n, max_size=n)
        t = text()
        return draw(c), draw(label), draw(x), draw(y), draw(t)

    @given(generate_good_Chart_data())
    def test_Chart(self, g):
        c, label, x, y, t = g
        c.add_DataSet(label, x, y)
        c.edit_title(t)
        c.edit_xAxes(t)
        c.edit_yAxes(t, "linear")
        assert c.title["text"] == t
        assert c.xAxes[0]["scaleLabel"]["labelString"] == t
        assert c.yAxes[0]["scaleLabel"]["labelString"] == t

    @composite
    def generate_bad_ScatterChart_data(draw):
        n = draw(integers(0, 50))
        label = text()
        x = iterables(one_of(integers(), floats()), min_size=n + 1, max_size=n + 1)
        y = iterables(one_of(integers(), floats()), min_size=n, max_size=n)
        return draw(label), draw(x), draw(y)

    @given(generate_bad_ScatterChart_data())
    def test_ScatterChart_xy_Exceptions(self, g):
        label, x, y = g
        l = ScatterChart()
        with pytest.raises(Exception):
            l.add_DataSet(label, x, y)

    @given(generate_bad_ScatterChart_data())
    def test_ScatterChart_xu_Exception(self, g):
        label, x, y = g
        l = ScatterChart()
        with pytest.raises(Exception):
            l.add_DataSet(label, x, x, y)

    @composite
    def generate_bad_Chart_data(draw):
        n = draw(integers(0, 50))
        label = text()
        c = one_of(builds(BarChart), builds(LineChart))
        x = iterables(one_of(integers(), floats()), min_size=n + 1, max_size=n + 1)
        y = iterables(one_of(integers(), floats()), min_size=n, max_size=n)
        return draw(c), draw(label), draw(x), draw(y)

    @given(generate_bad_Chart_data())
    def test_LineBarChart_label_Exception(self, g):
        c, label, x, y = g
        with pytest.raises(Exception):
            l.add_DataSet(label, x, x)
            l.add_DataSet(label, y, y)
