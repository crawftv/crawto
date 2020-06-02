import IPython
import pytest
from hypothesis import given
from hypothesis.strategies import (builds, composite, floats, integers,
                                   iterables, one_of, text)
from IPython.display import HTML, display

from crawto.charts.charts import (BarChart, Chart_type, LineChart, Plot,
                                  ScatterChart)


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
        l = ScatterChart()
        with pytest.raises(Exception):
            label, x, y = g
            l.add_DataSet(label, x, y)

    @given(generate_bad_ScatterChart_data())
    def test_ScatterChart_xu_Exception(self, g):
        l = ScatterChart()
        with pytest.raises(Exception):
            label, x, y = g
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
        with pytest.raises(Exception):
            c, label, x, y = g
            l.add_DataSet(label, x, x)
            l.add_DataSet(label, y, y)
