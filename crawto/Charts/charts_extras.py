from IPython.display import display, HTML
from string import Template
import json
from typing import List
from ..Charts.chart_type import Data
import jsons
import matplotlib.pyplot as plt
import seaborn as sns


def scatter_plot(html_data) -> HTML:

    html = make_html(html_data)
    return HTML(html)


def make_scatter_html(data: Data) -> str:

    html = Template(
        """
            <head>
                <script type="application/javascript" src="https:\\cdnjs.cloudflare.com/ajax/libs/require.js/x.y.z/require.js"></script>
            </head>
            <body>
            <h1> T-sne </h1>
            <canvas id="t-sne-chart"></canvas>
            <script> $js_text </script>
            </body>
            """
    )

    js_text_template = Template(
        """
            requirejs(['https:\\cdnjs.cloudflare.com/ajax/libs/Chart.js/2.8.0/Chart.js'], function(Chart){
                new Chart(document.getElementById("t-sne-chart"), {
                        type: "scatter",
                        data: $data,
                        options: {
                            tooltips: {
                                callbacks: {
                                    label: function(tooltipItem, data) {
                                        var label = data.datasets[tooltipItem.datasetIndex].label;
                                        return label
                                   }
                                }
                            }
                        }
                    });
                });     
            """
    )
    js_text = js_text_template.substitute({"data": jsons.dumps(data)})
    html = html.substitute({"js_text": js_text})

    return html


def feature_importances_plot(columns, feature_importances):
    d = list(zip(columns, feature_importances))
    d.sort(key=lambda tup: tup[1])
    d = d[::-1]
    x = [i[0] for i in d]
    y = [i[1] for i in d]
    g = sns.barplot(x=x, y=y)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
