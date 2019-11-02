import uuid

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

    def add_colors(self):
        l = len(self.colorscheme)
        for i, j in enumerate(self.data["datasets"]):
            j["backgroundColor"] = self.colorscheme[i % l]
            j["borderColor"] = self.colorscheme[i % l]
        return self.data

    @property
    def html(self):
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
        self.add_colors()
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
        new_dict = {"display": "true", "text": text}
        self.title.update(new_dict)

    def edit_xAxes(self, text: str = "", axisIndex: int = 0):
        self.xAxes[axisIndex]["scaleLabel"]["display"] = "true"
        self.xAxes[axisIndex]["scaleLabel"]["labelString"] = text

    def edit_yAxes(self, text: str = "", type: str = "linear", axisIndex: int = 0):
        self.yAxes[axisIndex]["scaleLabel"]["display"] = "true"
        self.yAxes[axisIndex]["scaleLabel"]["labelString"] = text
        self.yAxes[axisIndex]["type"] = type

    def __repr__(self):
        return self.html


class ScatterChart(Chart):
    @property
    def type(self):
        return "scatter"

    def add_DataSet(self, label: str, x, y, unique_identifier=None):
        x, y, = list(x), list(y)
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
    @property
    def type(self):
        return "bar"

    def add_DataSet(self, label: str, x, y):
        if "labels" not in self.data.keys():
            self.data["labels"] = list([str(i) for i in x])
        elif list([str(i) for i in x]) != self.data["labels"]:
            raise Exception(f"Already defined the labels for this chart")

        self.data["datasets"].append({"label": label, "data": list(y)})


class LineChart(Chart):
    @property
    def type(self):
        return "line"

    def add_DataSet(self, label: str, x, y, fill: str = "false"):
        if "labels" not in self.data.keys():
            self.data["labels"] = list([str(i) for i in x])
        elif list([str(i) for i in x]) != self.data["labels"]:
            raise Exception(f"Already defined the labels for this chart")

        self.data["datasets"].append({"label": label, "data": list(y), "fill": fill})
