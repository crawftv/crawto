import pytest
import IPython
from IPython.display import display, HTML
from crawto.charts.charts import Plot, ScatterChart, BarChart, LineChart
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

l2 = LineChart()

#@given(label = text(),
#        x=iterables(one_of(integers(),floats())),
#        y=iterables(one_of(integers(), floats())),
#        t = text())
@composite
def generate_LineChart_data(draw):
    n = draw(integers(0,5000))
    label = text()
    x=iterables(one_of(integers(),floats()),min_size=n,max_size=n)
    y=iterables(one_of(integers(),floats()),min_size=n,max_size=n)
    t = text()
    return draw(label),draw(x),draw(y),draw(t)
@given(generate_LineChart_data())
def test_LineChart(g):
    label,x,y,t = g
    l2.add_DataSet(label,x,y)
    l2.edit_title(t)
    l2.edit_xAxes(t)
    l2.edit_yAxes(t,"linear")

