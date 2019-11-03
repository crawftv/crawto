from IPython.display import display, HTML
from string import Template
import json
from typing import List
import jsons
import matplotlib.pyplot as plt
import seaborn as sns


def feature_importances_plot(columns, feature_importances):
    d = list(zip(columns, feature_importances))
    d.sort(key=lambda tup: tup[1])
    d = d[::-1]
    x = [i[0] for i in d]
    y = [i[1] for i in d]
    g = sns.barplot(x=x, y=y)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
