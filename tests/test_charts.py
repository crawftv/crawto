import pytest
import IPython
from IPython.display import display, HTML
from crawto.charts.charts import Plot, ScatterChart, BarChart, LineChart, Chart_type
from hypothesis import given
from hypothesis.strategies import text, builds, iterables, floats, one_of, integers,composite

#Simple Test
x=[-1,0,1]
y = [1,-1,1]

l = LineChart()
l.add_DataSet("t", x, y)
l.add_DataSet("t1" ,x, [2,2,2])
l.edit_title("LineChart")
l.edit_xAxes("x Axis")
l.edit_yAxes("y Axis","linear")

c = ScatterChart()
c.add_DataSet("test",x,y)
c.add_DataSet("t1" ,x,[2,2,2],["a","b","c"])
c.edit_title("ScatterChart")

b = BarChart()
b.add_DataSet("t",x,y)
b.add_DataSet("t1" ,x,[2,2,2])
b.edit_title("BarChart")


class TestChart:
    def __init__(self,chart:Chart_type):
        self.chart = chart
    @composite
    def generate_good_Chart_data(draw):
        n = draw(integers(0,50))
        label = text()
        x=iterables(one_of(integers(),floats()),min_size=n,max_size=n)
        y=iterables(one_of(integers(),floats()),min_size=n,max_size=n)
        t = text()
        return draw(label),draw(x),draw(y),draw(t)
    @given(generate_good_Chart_data())
    def test_Chart(g):
        c = self.chart
        label,x,y,t = g
        c.add_DataSet(label,x,y)
        c.edit_title(t)
        c.edit_xAxes(t)
        c.edit_yAxes(t,"linear")
        assert c.title["text"]==t
        assert c.xAxes[0]["scaleLabel"]["labelString"] == t
        assert c.yAxes[0]["scaleLabel"]["labelString"] == t
test_LineChart = TestChart(LineChart())
#class TestLineChart:
#    @composite
#    def generate_good_LineChart_data(draw):
#        n = draw(integers(0,50))
#        label = text()
#        x=iterables(one_of(integers(),floats()),min_size=n,max_size=n)
#        y=iterables(one_of(integers(),floats()),min_size=n,max_size=n)
#        t = text()
#        return draw(label),draw(x),draw(y),draw(t)
#    @given(generate_good_LineChart_data())
#    def test_LineChart(g):
#        l2 = LineChart()
#        label,x,y,t = g
#        l2.add_DataSet(label,x,y)
#        l2.edit_title(t)
#        l2.edit_xAxes(t)
#        l2.edit_yAxes(t,"linear")
#        assert l2.title["text"]==t
#        assert l2.xAxes[0]["scaleLabel"]["labelString"] = t
#        assert l2.yAxes[0]["scaleLabel"]["labelString"] = t
#    def test_LineChart_exception():
#        l2 = LineChart
#
#class TestBarChart:
#    @composite
#    def generate_good_BarChart_data(draw):
#        n = draw(integers(0,50))
#        label = text()
#        x=iterables(one_of(integers(),floats()),min_size=n,max_size=n)
#        y=iterables(one_of(integers(),floats()),min_size=n,max_size=n)
#        t = text()
#        return draw(label),draw(x),draw(y),draw(t)
#    @given(generate_good_BarChart_data())
#    def test_BarChart(g):
#        b2 = BarChart()
#        label,x,y,t = g
#        b2.add_DataSet(label,x,y)
#        b2.edit_title(t)
#        b2.edit_xAxes(t)
#        b2.edit_yAxes(t,"linear")
#        assert b2.title["text"]==t
#        assert b2.xAxes[0]["scaleLabel"]["labelString"] = t
#        assert b2.yAxes[0]["scaleLabel"]["labelString"] = t
