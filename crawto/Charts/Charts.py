import json
import jsons
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

Chart = Union[ScatterChart, BarChart, LineChart]


class Chart:
    """Base class for the Bar, Line, & Scatter Charts.

    Parameters
    ----------
    data : Dict
        The data for the chart. Needs to be served in a specific way for each chart.

    title : str, default = None

    colorscheme : List, default = The tableau classic 10.



    Attributes
    ----------
    xAxes : Dict
        A list of xAxes for the chart. Default is length 1.
    yAxes
        A list of yAxes for the chart. Default is length 1.

    Examples
    --------

    >>>from crawto import CrawtoDoc.py
    >>>edit_title("Title")
    >>>edit_xAxes("X Axis", 0)
    >>>edit_yAxes("Y Axis", "linear", 0)
    >>>html()

    See also
    --------


    References
    ----------

    """

    def __init__(
        self,
        data: Dict[str, List] = None,
        title: Dict[str, str] = None,
        colorscheme: List[str] = None,
    ):
        self.id = uuid.uuid1()
        self.data = data if data is not None else {"datasets": []}
        self.colorscheme = "tableauClassic10"
        self.title = title if title is not None else {"display": "false", "text": ""}
        self.colorscheme = (
            colorscheme if colorscheme is not None else default_colorscheme
        )
        self.xAxes = [
            {"display": "true", "scaleLabel": {"display": "false", "labelString": ""}}
        ]
        self.yAxes = [
            {
                "display": "true",
                "scaleLabel": {"display": "false", "labelString": ""},
                "type": "linear",
            }
        ]

    def _add_colors(self):
        """Iterates over the data and adds the color value

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
        """Updates the HTML on change to any of the elements of the chart.

        Parameters
        ----------

        self

        Returns
        -------
        html : string
           returns html to be rendered by IPython
        """
        html = Template(
            """
        <canvas id= "$id" ></canvas>
        <script>
        requirejs(['https:\\cdnjs.cloudflare.com/ajax/libs/Chart.js/2.8.0/Chart.js'], function(Chart){
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
        """
        )
        self._add_colors()
        html = html.substitute(
            {
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
        """Adds or updates the title for the chart.

    Parameters
    ----------
    text : string
        This string will be the new title

    Returns
    -------
    Does not return anything, updates self.title
    """
        new_dict = {"display": "true", "text": text}
        self.title.update(new_dict)

    def edit_xAxes(self, text: str = "", axisIndex: int = 0):
        """Edit the xAxes for the chart.

        Parameters
        ----------
        text : <class 'str'>, default =""
            Text to display
        axisIndex : <class 'int'>, default = 0
            Chartjs requires the axes to be in an array, even if there is only one. Defaulted at zero.

        Returns
        -------
        """
        self.xAxes[axisIndex]["scaleLabel"]["display"] = "true"
        self.xAxes[axisIndex]["scaleLabel"]["labelString"] = text

    def edit_yAxes(self, text: str = "", type: str = "linear", axisIndex: int = 0):
        """Edit the y Axis

        Parameters
        ----------
        text : <class 'str'>, default =
            Text to display

        type : <class 'str'>, default = linear
            The scale of the axis. Options are linear, category, logarithmic, time

        axisIndex : <class 'int'>, default = 0
            Chartjs requires the axes to be in an array, even if there is only one. Defaulted at zero.

        Returns
        -------
        """
        self.yAxes[axisIndex]["scaleLabel"]["display"] = "true"
        self.yAxes[axisIndex]["scaleLabel"]["labelString"] = text
        self.yAxes[axisIndex]["type"] = type

    def __repr__(self):
        return self.html


class ScatterChart(Chart):
    """Creates a scatter chart.

    Attributes
    ----------

    Examples
    --------

    >>>from crawto.Chart.Charts import ScatterChart
    >>>ScatterChart()
    >>>html()
    >>>edit_title(text)
    >>>edit_xAxes(text, axisIndex)
    >>>edit_yAxes(text, type, axisIndex)

    See also
    --------


    References
    ----------

    """

    @property
    def type(self):
        """type updates the html attribute from the base class
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

    Attributes
    ----------


    Examples
    --------

    >>>from crawto.Charts.Chart import BarChart
    >>>BarChart()
    >>>html()

    See also
    --------


    References
    ----------

    """

    @property
    def type(self) -> str:
        """type updates the html attribute from the base class
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
            self.data["labels"] = list([str(i) for i in x])
        elif list([str(i) for i in x]) != self.data["labels"]:
            raise Exception(f"Already defined the labels for this chart")
        y = [int(i) for i in y]
        self.data["datasets"].append({"label": label, "data": y})


class LineChart(Chart):
    """Generates a line chart.

    Attributes
    ----------


    Examples
    --------

    >>>from crawto.Charts.Chart import LineChart
    >>>LineChart(data, title, colorscheme)
    >>>l.add_Dataset("t1",[-1,0,1],[1,-1,1],fill="false")
    >>>l.html()

    See also
    --------

    References
    ----------

    """

    @property
    def type(self):
        """type updates the html attribute from the base class
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
            an array of the heights of each bar.

        fill : str, default = 'false'
             javscript boolean for whether or not fill the area underneath the line.
             "true" or "false"

        Returns
        -------
        """
        if "labels" not in self.data.keys():
            self.data["labels"] = list([str(i) for i in x])
        elif list([str(i) for i in x]) != self.data["labels"]:
            raise Exception(f"Already defined the labels for this chart")
        self.data["datasets"].append({"label": label, "data": list(y), "fill": fill})


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

    >>>from crawto import CrawtoDoc.py
    >>>Plot()
    >>>l = LineChart()
    >>>l.add_Dataset("t1",[-1,0,1],[1,-1,1])
    >>>add_column(l, "sixteen")
    >>>display()

    See also
    --------


    References
    ----------

    """

    def __init__(self):
        self.head = """
        <head>
                <script type="application/javascript"
                src="https:\\cdnjs.cloudflare.com/ajax/libs/require.js/x.y.z/require.js"></script>
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css">
                <script src="https://code.jquery.com/jquery-3.1.1.min.js"
                  integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8="
                  crossorigin="anonymous"></script>
                <script src="https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.js"></script>
        </head>
        """
        self.body = """
        <body>
        <div class= "ui grid">
        """

    def add_column(self, chart: Chart, width: str = "sixteen"):
        """Adds a chart to the plot area. Column refers to the SemanticUI api.
        Total width per row is 16. When the width of multiple columns combines
        to be greater than sixteen, a new row is created.

        Parameters
        ----------

        chart : Chart
            The chart to be plotted. Must be a Chart type.
        width : str, default = "sixteen"
            The width of the chart. "sixteen" is the largest, "one" is the smallest.

        Returns
        -------
        """
        html = f"<div class = '{width} wide column'>\n"
        html += chart.html + "\n"
        html += "\n</div>\n"
        self.body += html

    def display(self):
        """The function to call to actually display the plot.

        Parameters
        ----------

        Returns
        -------
        Renders the HTML
        """
        d = self.head + self.body + "\n</div>\n</body>"
        return HTML(d)
