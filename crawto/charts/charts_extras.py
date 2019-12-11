from IPython.display import display, HTML
from string import Template
import json
from typing import List
import jsons
import matplotlib.pyplot as plt
import seaborn as sns
from .charts import ScatterChart, Plot,BarChart
from statsmodels.api import ProbPlot
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error


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
    return s


def residuals_vs_target_chart(
    target_values, residuals, unique_identifier, width="eight"
):
    s = ScatterChart(width=width)
    s.add_DataSet(
        "Residuals vs. target Values", target_values, residuals, unique_identifier
    )
    s.edit_xAxes("target Values")
    s.edit_yAxes("Residuals")
    return s


def pp_plot(theoretical_percentiles, sample_percentiles, unique_identifier=None, width="eight",):
    s = ScatterChart(width=width)
    s.add_DataSet("PP-Plot", theoretical_percentiles, sample_percentiles)
    s.edit_xAxes("Theoretical Probabilities")
    s.edit_yAxes("Sample Probabilities")
    return s


def qq_plot(theoretical_percentiles, sample_percentiles, unique_identifier=None, width="eight"):
    s = ScatterChart(width=width)
    s.add_DataSet("QQ-Plot", theoretical_percentiles, sample_percentiles)
    s.edit_xAxes("Theoretical Probabilities")
    s.edit_yAxes("Sample Probabilities")
    return s

def coefficient_plot(top_coefs):
    coefs = [i[0] for i in top_coefs]
    y = [i[1][1] for i in top_coefs]
    b = BarChart()
    b.add_DataSet("Model Coefficients",coefs, y)
    b.edit_yAxes("Coefficient Values")
    return b
def regression_viz(y_pred, y_true, index,top_coefs):

    residuals = (y_pred - y_true)
    rvp = residuals_vs_predicted_chart(y_pred, residuals, index)
    rvf = residuals_vs_target_chart(y_true, residuals, index)
    pp = ProbPlot(residuals)
    ppplot = pp_plot(pp.theoretical_percentiles, pp.sample_percentiles.ravel())
    qqplot = qq_plot(pp.theoretical_quantiles, pp.sample_quantiles.ravel())
    evs = round(explained_variance_score(y_true, y_pred),2)
    r2 = round(r2_score(y_true, y_pred),2)
    mse = round(mean_squared_error(y_true, y_pred),2)
    coef_plot = coefficient_plot(top_coefs)
    p = Plot()
    p.add_column(rvp)
    p.add_column(rvf)
    p.add_column(ppplot)
    p.add_column(qqplot)
    p.add_column(coef_plot)
    p.top = regression_stats(mse,r2,evs)
    return p


def regression_stats(mse, r2, evs):
    rs = Template("""
    <div class="ui statistics">
    <div class="statistic">
        <div class="value">
        $mse
        </div>
        <div class="label">
        Mean Squared Error
        </div>
    </div>
    <div class="statistic">
        <div class="value">
        $r2
        </div>
        <div class="label">
        R-Squared Score
        </div>
    </div>
    <div class="statistic">
        <div class="value">
        $evs
        </div>
        <div class="label">
        Explained Variance Score
        </div>
    </div>
    </div>
     """)
    stats = rs.substitute({"mse":mse,"r2":r2,"evs":evs})
    return stats
