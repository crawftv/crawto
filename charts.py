from IPython.display import display, HTML
from string import Template
import json


def tsne(label_list, x_list, y_list):
    datasets = [
        {"x": round(float(x), 2), "y": round(float(y), 2)}
        for x, y in list(zip(x_list, y_list))
    ]
    label_list = list(label_list)
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
                    data: {
                        labels: $labels,
                        datasets: [{
                            label: "Latin Data",
                            data: $data,
                        }],
                    },
                    options: {
                        tooltips: {
                            callbacks: {
                                label: function(tooltipItem, data) {
                                    var label = data.labels[tooltipItem.index];
                                    return label + ': (' + tooltipItem.xLabel + ', ' + tooltipItem.yLabel + ')';
                                }
                            }
                        }
                    }
                });
            });     
        """
    )
    js_text = js_text_template.substitute(
        {"data": json.dumps(datasets), "labels": json.dumps(label_list)}
    )
    html = html.substitute({"js_text": js_text})

    return HTML(html)
