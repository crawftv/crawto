from IPython.display import display, HTML
from string import Template
import numpy as np
import json
from typing import List
import jsons
import matplotlib.pyplot as plt
import seaborn as sns
from .charts import ScatterChart, Plot, BarChart, LineChart
from statsmodels.api import ProbPlot
from sklearn.metrics import (
    explained_variance_score,
    r2_score,
    mean_squared_error,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.multiclass import unique_labels


def feature_importances_plot(columns, feature_importances):
    d = list(zip(columns, feature_importances))
    d.sort(key=lambda tup: tup[1])
    d = d[::-1]
    x = [i[0] for i in d]
    y = [i[1] for i in d]
    g = sns.barplot(x=x, y=y)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)


def residuals_vs_predicted_chart(
    predicted_values, residuals, unique_identifier, width="eight"
):
    s = ScatterChart(width=width)
    s.add_DataSet(
        "Residuals vs. Predicted Values", predicted_values, residuals, unique_identifier
    )
    s.edit_xAxes("Predicted Values")
    s.edit_yAxes("Residuals")
    s.edit_title("Residuals vs. Predicted Values")
    return s


def residuals_vs_target_chart(
    target_values, residuals, unique_identifier, width="eight"
):
    s = ScatterChart(width=width)
    s.add_DataSet(
        "Residuals vs. Target Values", target_values, residuals, unique_identifier
    )
    s.edit_xAxes("target Values")
    s.edit_yAxes("Residuals")
    s.edit_title("Residuals vs. Target Values")
    return s


def pp_plot(
    theoretical_percentiles, sample_percentiles, unique_identifier=None, width="eight",
):
    s = ScatterChart(width=width)
    s.add_DataSet("PP-Plot", theoretical_percentiles, sample_percentiles)
    s.edit_xAxes("Theoretical Probabilities")
    s.edit_title("PP Plot")
    s.edit_yAxes("Sample Probabilities")
    return s


def qq_plot(
    theoretical_percentiles, sample_percentiles, unique_identifier=None, width="eight"
):
    s = ScatterChart(width=width)
    s.add_DataSet("QQ-Plot", theoretical_percentiles, sample_percentiles)
    s.edit_xAxes("Theoretical Probabilities")
    s.edit_yAxes("Sample Probabilities")
    s.edit_title("QQ Plot")
    return s


def coefficient_plot(top_coefs):
    coefs = [i[0] for i in top_coefs]
    y = [i[1][1] for i in top_coefs]
    b = BarChart()
    b.add_DataSet("Model Coefficients", coefs, y)
    b.edit_title("Largest Coefficients")
    b.edit_yAxes("Coefficient Values")
    return b


def regression_viz(y_pred, y_true, index, top_coefs):

    residuals = y_pred - y_true
    rvp = residuals_vs_predicted_chart(y_pred, residuals, index)
    rvf = residuals_vs_target_chart(y_true, residuals, index)
    pp = ProbPlot(residuals)
    ppplot = pp_plot(pp.theoretical_percentiles, pp.sample_percentiles.ravel())
    qqplot = qq_plot(pp.theoretical_quantiles, pp.sample_quantiles.ravel())
    evs = round(explained_variance_score(y_true, y_pred), 2)
    r2 = round(r2_score(y_true, y_pred), 2)
    mse = round(mean_squared_error(y_true, y_pred), 2)
    coef_plot = coefficient_plot(top_coefs)
    p = Plot()
    p.add_column(rvp)
    p.add_column(rvf)
    p.add_column(ppplot)
    p.add_column(qqplot)
    p.add_column(coef_plot)
    p.top = viz_stats(
        {"Mean Squared Error": mse, "R-Squared": r2, "Explained Variance Score": evs}
    )

    return p


def viz_stats(stats_dict):
    s = ""
    for key, value in stats_dict.items():
        s += f"""
        <div class="statistic">
        <div class="value">
        {value}
        </div>
        <div class="label">
        {key}
        </div>
        </div>
        """

    rs = Template(
        """
    <div class="ui statistics">
        $s
    </div>
     """
    )
    stats = rs.substitute({"s": s})
    return stats


def classification_viz(y_true, y_pred, y_pred_proba):
    roc = roc_plot(y_true, y_pred_proba)
    prc = prc_plot(y_true, y_pred_proba)
    cr = ClassificationReport(y_true, y_pred)
    cm = ConfusionMatrix(y_true, y_pred, labels=[0, 1])
    ras = round(roc_auc_score(y_true, y_pred) * 100, 2)
    aps = round(average_precision_score(y_true, y_pred) * 100, 2)
    p = Plot()
    p.add_column(cr)
    p.add_column(cm)
    p.add_column(roc)
    p.add_column(prc)
    p.top = viz_stats({"Average Precision Score": aps, "ROC AUC Score": ras})
    return p


def roc_plot(y_true, y_pred, width="eight"):
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    l = LineChart(width=width)
    l.add_DataSet("ROC Curve", fpr, tpr)
    l.add_DataSet("y=x", fpr, fpr)
    l.edit_xAxes("False Positive Rate")
    l.edit_yAxes("True Positive Rate")
    l.xAxes[0]["type"] = "linear"
    l.xAxes[0]["ticks"] = {"min": 0.0, "max": 1.0, "stepSize": 0.1}
    l.edit_title("ROC Plot")
    return l


def prc_plot(y_true, y_pred, width="eight"):
    a, b, c = precision_recall_curve(y_true, y_pred)
    l = LineChart(width=width)
    l.add_DataSet("Precision Recall Curve", a, b)
    x = list(np.linspace(0, 1, len(a)))
    y = x[::-1]
    l.add_DataSet("y=-x", x, y)
    l.edit_xAxes("False Positive Rate")
    l.edit_yAxes("True Positive Rate")
    l.xAxes[0]["type"] = "linear"
    l.xAxes[0]["ticks"] = {"min": 0.0, "max": 1.0, "stepSize": 0.1}
    l.edit_title("Precision Recall Curve")
    return l


class ClassificationReport:
    def __init__(self, y_true, y_pred, width="eight"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.width = width

    @property
    def cr_dict(self):
        cr_dict = classification_report(self.y_true, self.y_pred, output_dict=True)
        return cr_dict

    @property
    def html(self):
        cr = Template(
            """
        <div class= "$width wide column">
        <h2>Classification Report</h2>
        <table class="ui celled table">
          <thead>
            <tr>
            <th></th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
          </tr></thead>
          <tbody>
              $tr
          </tbody>
        </table>
        </div>"""
        )

        def create_tr(key, dict_element):
            tr = "<tr>"
            tr += f'<td data-label="">{key}</td>'
            tr += (
                f'<td data-label="Precision">{round(dict_element["precision"],2)}</td>'
            )
            tr += f'<td data-label="Precision">{round(dict_element["recall"],2)}</td>'
            tr += f'<td data-label="Precision">{round(dict_element["f1-score"],2)}</td>'
            tr += f'<td data-label="Precision">{round(dict_element["support"],2)}</td>'
            tr += "</tr>"
            return tr

        k = list(self.cr_dict.keys())
        k.remove("accuracy")
        tr = "".join([create_tr(i, self.cr_dict[i]) for i in k])
        cr = cr.substitute({"tr": tr, "width": self.width})
        return cr


class ConfusionMatrix:
    def __init__(self, y_true, y_pred, labels, width="eight"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels

        self.width = width

    @property
    def cm(self):
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
        return cm

    @property
    def header(self):
        return "<th></th>" + "".join([f" <th> {i} </th>" for i in self.h_labels])

    @property
    def h_labels(self):
        h_labels = [f"Predicted {i}" for i in self.labels]
        return h_labels

    @property
    def b_labels(self):
        b_labels = [f"Actual {i}" for i in self.labels]
        return b_labels

    @property
    def tr(self):
        tr = ""
        for i, j in enumerate(self.b_labels):
            tr += f'<tr><td data-label="">{j}</td>'
            for x, y in enumerate(self.h_labels):
                tr += f'<td data-label="{y}">{self.cm[i][x]}</td>'

            tr += "</tr>"
        return tr

    @property
    def html(self):
        cm = Template(
            """
        <div class= "$width wide column">
        <h2>Confusion Matrix</h2>
        <table class="ui celled table">
          <thead>
            <tr>
            $header
          </tr></thead>
          <tbody>
              $tr
          </tbody>
        </table>
        </div>"""
        )
        cm = cm.substitute({"tr": self.tr, "width": self.width, "header": self.header})
        return cm
