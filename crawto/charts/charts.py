import json
import jsons
from functools import reduce
from numpy import inf, NaN
import uuid
from IPython.display import display, HTML
from string import Template
from typing import Union, List, Dict

default_colorscheme = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]


class Chart:
    """
    Base class for the Bar, Line, & Scatter Charts.

    Parameters
    ----------
    data : Dict
        The data for the chart. Needs to be served in a specific way for each chart.

    title : str, default = None

    colorscheme : List, default = default_colorscheme

    width : str, default = "sixteen"
        The width of the column. The total space is 16, so "sixteen" is \
        the largest and "one" is the smallest.\
        The height of the column adjusts to width.



    Attributes
    ----------
    id : str, default = uuid1
        A unique id to link the chart function and canvas tag.
    xAxes : List[Dict]
        A list of xAxes for the chart. Default is length 1.
    yAxes : List[Dict]
        A list of yAxes for the chart. Default is length 1.

    Examples
    --------

    >>> edit_title("Title")
    >>> edit_xAxes("X Axis", 0)
    >>> edit_yAxes("Y Axis", "linear", 0)
    >>> html()
    """

    def __init__(
        self,
        data: Dict[str, List] = None,
        title: Dict[str, str] = None,
        colorscheme: List[str] = None,
        width: str = "sixteen",
    ):
        self.id = uuid.uuid1()
        self.data = data if data is not None else {"datasets": []}
        self.colorscheme = "tableauClassic10"
        self.title = title if title is not None else {"display": "false", "text": ""}
        self.colorscheme = (
            colorscheme if colorscheme is not None else default_colorscheme
        )
        self.width = width
        self.xAxes = [
            {"display": "true", "scaleLabel": {"display": "false", "labelString": ""},}
        ]
        self.yAxes = [
            {
                "display": "true",
                "scaleLabel": {"display": "false", "labelString": ""},
                "type": "linear",
            }
        ]

    def _add_colors(self):
        """
        Internal method. Iterates over the data and adds the color value from colorscheme to datasets.

        Parameters
        ----------

        Returns
        -------
        """
        l = len(self.colorscheme)
        for i, j in enumerate(self.data["datasets"]):
            j["backgroundColor"] = self.colorscheme[i % l]
            j["borderColor"] = self.colorscheme[i % l]

    @property
    def html(self) -> str:
        """
        Updates the HTML on change to any of the elements of the chart.

        Parameters
        ----------
        Returns
        -------
        html : str
           Returns html to be rendered by IPython
        """
        html = Template(
            """
        <div class = '$width wide column'>
        <canvas id= "$id" width="1" height="1" ></canvas>
        <script>
        requirejs(['https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.8.0/Chart.js'], function(Chart){
            new Chart(document.getElementById("$id"), {
                    type: "$type",
                    data: $data,
                    options: {
                        "responsive": true,
                        "title": $title,
                        "scales" : { xAxes : $xAxes,
                                     yAxes : $yAxes
                                     },
                        tooltips: {
                            callbacks: {
                                label: function(tooltipItem, data) {
                                    var label = data.datasets[tooltipItem.datasetIndex].label;
                                    var word = data.datasets[tooltipItem.datasetIndex].data[tooltipItem.index].Value
                                    return word == undefined? label : label + " : " + word
                               }
                            }
                        }
                    }
                });
                });
        </script>
        </div>
        """
        )
        self._add_colors()
        html = html.substitute(
            {
                "width": self.width,
                "data": jsons.dumps(self.data),
                "id": self.id,
                "title": jsons.dumps(self.title),
                "xAxes": jsons.dumps(self.xAxes),
                "yAxes": jsons.dumps(self.yAxes),
                "type": self.type,
            }
        )
        return html

    def edit_title(self, text: str):
        """
        Adds or updates the title for the chart.

        Parameters
        ----------
        text : str
            This string will be the new title

        Returns
        -------
        Does not return anything, updates self.title
        """
        new_dict = {"display": "true", "text": text}
        self.title.update(new_dict)

    def edit_xAxes(self, text: str = "", axisIndex: int = 0):
        """
        Edit the xAxes for the chart.

        Parameters
        ----------
        text : str, default =""
            Text to display
        axisIndex : int, default = 0
            Chartjs requires the axes to be in an array, even if there is only one. Defaulted at zero.

        Returns
        -------
        """
        self.xAxes[axisIndex]["scaleLabel"]["display"] = "true"
        self.xAxes[axisIndex]["scaleLabel"]["labelString"] = text

    def edit_yAxes(self, text: str = "", type: str = "linear", axisIndex: int = 0):
        """
        Edit the y Axis

        Parameters
        ----------
        text : str, default =
            Text to display

        type : str, default = linear
            The scale of the axis. Options are linear, category, logarithmic, time

        axisIndex : int, default = 0
            Chartjs requires the axes to be in an array, even if there is only one. Defaulted at zero.

        Returns
        -------
        """
        self.yAxes[axisIndex]["scaleLabel"]["display"] = "true"
        self.yAxes[axisIndex]["scaleLabel"]["labelString"] = text
        self.yAxes[axisIndex]["type"] = type


class ScatterChart(Chart):
    """Creates a scatter chart.

    Parameters
    ----------
    data : Dict
        The data for the chart. Needs to be served in a specific way for each chart.

    title : str, default = None

    colorscheme : List, default = The tableau classic 10.

    Attributes
    ----------
    id : str , default = uuid1
        A unique id to link the chart function and canvas tag.
    xAxes : Dict
        A list of xAxes for the chart. Default is length 1.
    yAxes
        A list of yAxes for the chart. Default is length 1.

    Examples
    --------
    >>> from crawto.charts.charts import ScatterChart
    >>> s = ScatterChart()
    >>> s.add_DataSet("ScatterChart",[1,1,1],[2,2,2])
    >>> s.edit_title("Title")
    >>> s.edit_xAxes("X Axis", 0)
    >>> s.edit_yAxes("Y Axis", "linear", 0)
    >>> s.html()

    """

    @property
    def type(self):
        """Type updates the html attribute from the base class
        """
        return "scatter"

    def add_DataSet(self, label: str, x, y, unique_identifier=None):
        """Adds a new dataset to the chart object.

        Parameters
        ----------

        label : array
            the label for the dataset

        x : array
           an array of the values for the x Axis. Can be numbers or Strings.

        y : array
            an array of the heights of each bar.


        unique_identifier : array, default = None
            an array of unique identifiers for each data point.

        Returns
        -------
        """
        x, y = list(x), list(y)
        u = list(unique_identifier) if unique_identifier is not None else None
        if len(x) != len(y):
            raise Exception("x and y columns are not equal in length")
        d = {
            "label": label,
            "data": [{"x": float(x[i]), "y": float(y[i])} for i in range(len(x))],
        }
        if u is not None:
            if len(u) != len(y):
                raise Exception(
                    "unique_identifier and x or y columns are not equal in length"
                )
            else:
                for i, j in enumerate(d["data"]):
                    j["Value"] = u[i]
        self.data["datasets"].append(d)


class BarChart(Chart):
    """Generates the html for the bar chart.

    Parameters
    ----------
    data : Dict
        The data for the chart. Needs to be served in a specific way for each chart.

    title : str, default = None

    colorscheme : List, default = The tableau classic 10.



    Attributes
    ----------
    id : str , default = uuid1
        A unique id to link the chart function and canvas tag.
    xAxes : List[Dict]
        A list of xAxes for the chart. Default is length 1.
    yAxes : List[Dict]
        A list of yAxes for the chart. Default is length 1.

    Examples
    --------
    >>> from crawto.charts.charts import BarChart
    >>> b = BarChart()
    >>> b.add_DataSet("BarChart",[1,1,1],[2,2,2])
    >>> b.edit_title("Title")
    >>> b.edit_xAxes("X Axis", 0)
    >>> b.edit_yAxes("Y Axis", "linear", 0)
    >>> b.html()

    """

    @property
    def type(self) -> str:
        """Type updates the html attribute from the base class
        """
        return "bar"

    def add_DataSet(self, label: str, x, y):
        """Adds a dataset to the chart.

        Parameters
        ----------
        label : array
            the label for the dataset
        x : array
           an array of the values for the x Axis. Can be numbers or Strings.

        y : array
            an array of the heights of each bar.

        Returns
        -------
        """
        if "labels" not in self.data.keys():
            self.data["labels"] = [str(i) for i in x]
        elif [str(i) for i in x] != self.data["labels"]:
            raise Exception(f"Already defined the labels for this chart")
        y = [float(i) for i in y if i not in [inf, -inf, NaN]]
        self.data["datasets"].append({"label": label, "data": y})


class LineChart(Chart):
    """Generates a line chart.

    Parameters
    ----------
    data : Dict
        The data for the chart. Needs to be served in a specific way for each chart.

    title : str, default = None

    colorscheme : List, default = The tableau classic 10.



    Attributes
    ----------
    id : str , default = uuid1
        A unique id to link the chart function and canvas tag.
    xAxes : List[Dict]
        A list of xAxes for the chart. Default is length 1.
    yAxes : List[Dict]
        A list of yAxes for the chart. Default is length 1.

    Examples
    --------
    >>> from crawto.charts.charts import LineChart
    >>> l = LineChart()
    >>> l.add_DataSet("LineChart",[1,1,1],[2,2,2])
    >>> l.edit_title("Title")
    >>> l.edit_xAxes("X Axis", 0)
    >>> l.edit_yAxes("Y Axis", "linear", 0)
    >>> l.html()

    """

    @property
    def type(self):
        """Type updates the html attribute from the base class
        """
        return "line"

    def add_DataSet(self, label: str, x, y, fill: str = "false"):
        """Adds a new dataset to the chart object.

        Parameters
        ----------
        label : array
            the label for the dataset
        x : array
           an array of the values for the x Axis. Can be numbers or Strings.

        y : array
            an array of the heights of each line.

        fill : str, default = 'false'
             javscript boolean for whether or not fill the area underneath the line.
             "true" or "false"

        Returns
        -------
        """
        #        if "labels" not in self.data.keys():
        #            self.data["labels"] = list([str(i) for i in x])
        #        elif list([str(i) for i in x]) != self.data["labels"]:
        #            raise Exception(f"Already defined the labels for this chart")
        #        y = [float(i) for i in y if i not in [inf,-inf,NaN]]
        #        self.data["datasets"].append({"label": label, "data": list(y), "fill": fill})
        x, y = list(x), list(y)
        if len(x) != len(y):
            raise Exception("x and y columns are not equal in length")
        d = {
            "label": label,
            "data": [{"x": float(x[i]), "y": float(y[i])} for i in range(len(x))],
            "fill": fill,
        }
        self.data["datasets"].append(d)


Chart_type = Union[ScatterChart, BarChart, LineChart]


class Plot:
    """Aggregates the Charts and renders the HTML

    Attributes
    ----------
    head : str
        The HTML header tag. Loads all the appropriate js.
    body : str
        The HTML body tags

    Examples
    --------

    >>> from crawto import CrawtoDoc.py
    >>> p=Plot()
    >>> l = LineChart()
    >>> l.add_Dataset("t1",[-1,0,1],[1,-1,1])
    >>> p.add_column(l, "sixteen")
    >>> p.display()

    """

    def __init__(self, columns: List[Chart_type] = None):
        self.head = """
        <head>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css">
                <script src="https://code.jquery.com/jquery-3.1.1.min.js"
                  integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8="
                  crossorigin="anonymous"></script>
                <script src="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.js"></script>
                <script>require.config({
                    shim: {
                        'chartjs': {
                            deps: ['moment']    // enforce moment to be loaded before chartjs
                        }
                    },
                    paths: {
                    'chartjs': 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.8.0/Chart.min.js',
                    'moment': 'https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.24.0/moment.min.js'
                    }});
                    </script>
        </head>
        """
        self.top = ""
        self.columns = columns if columns is not None else []

    def add_column(self, chart: Chart_type):
        return self.columns.append(chart)

    @property
    def body(self):
        columns = [i.html for i in self.columns]
        columns = reduce(lambda x, y: x + y, columns)
        body = Template(
            """
        <body>
            $top
            <div class= "ui grid">
        $columns
            </div>
        </body>
        """
        )
        body = body.substitute({"columns": columns, "top": self.top})
        return body

    @property
    def HTML(self):
        return self.head + self.body

    @property
    def display(self):
        """The function to call to actually display the plot.

        Parameters
        ----------

        Returns
        -------
        Renders the HTML
        """
        return HTML(self.HTML)
