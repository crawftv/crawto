import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import cloudpickle
import matplotlib.pyplot as plt
import pandas as pd
import papermill
import seaborn as sns
from pandas import DataFrame
from scipy.stats import probplot


@dataclass
class FeatureList:
    category: str
    transformation: str
    feature_pickle: List = field(default_factory=list, repr=False)

    @property
    def features(self):
        return cloudpickle.loads(self.feature_pickle)


@dataclass
class Cell:
    cell_type: str = "code"
    execution_count: int = None
    metadata: Dict = field(default_factory=dict)
    outputs: List = field(default_factory=list)
    source: List = field(default_factory=list)

    def add(self, line: str):
        line += "\n"
        self.source.append(line)
        return self


@dataclass
class Notebook:
    cells: List[Cell]
    metadata: Dict
    nbformat: int
    nbformat_minor: int


# create Cells


def create_import_cell(db_name, problem):
    import_cell = Cell()
    import_cell.add("import crawto.ml_analysis as ca")
    import_cell.add("import pandas as pd")
    import_cell.add(f"""db_name = "{db_name}" """)
    import_cell.add(f"""problem = "{problem}" """)
    import_cell.add(f"""target = "{target}" """)
    return import_cell


def create_feature_list_cell():
    cell = Cell()
    cell.source.append(
        "numeric_features, categoric_features = ca.get_feature_lists(db_name)"
    )
    return cell


def create_notebook(csv: str, problem: str, target: str, db_name: str):
    import_cell = asdict(create_import_cell(db_name=db_name, problem=problem))
    load_df = asdict(Cell().add(f"df = pd.read_csv('{csv}')"))
    na_report = asdict(Cell().add("ca.nan_report(df)"))
    skew_report = asdict(Cell().add("ca.skew_report(df)"))
    feature_list = asdict(create_feature_list_cell())
    correlation_report = asdict(
        Cell().add("ca.correlation_report(df,numeric_features,db_name)")
    )
    target_report = asdict(
        Cell().add("ca.target_distribution_report(problem=problem,df=df,target=target)")
    )
    probability_plot = asdict(
        Cell().add("ca.probability_plots(numeric_features,db_name,df)")
    )
    cells = [
        import_cell,
        load_df,
        na_report,
        skew_report,
        feature_list,
        correlation_report,
        target_report,
        probability_plot,
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.2",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open("crawto.ipynb", "w") as f:
        json.dump(nb, f)


def run_notebook():
    papermill.execute_notebook("crawto.ipynb", "crawto.ipynb")


# functions


def get_feature_lists(db_name):
    with sqlite3.connect(db_name) as conn:
        conn.row_factory = lambda c, r: FeatureList(*r)
        numeric_features = conn.execute(
            "SELECT * FROM features where category = 'numeric'"
        ).fetchall()
        categoric_features = conn.execute(
            "SELECT * FROM features where category = 'categoric'"
        ).fetchall()
    return numeric_features, categoric_features


def data_lookup(f: FeatureList, db_name: str, df: DataFrame) -> DataFrame:
    data_lookup_dict = {
        "imputed": "imputed_train_df",
        "transformed": "transformed_train_df",
        "untransformed": None,
    }
    if not data_lookup_dict[f.transformation]:
        df = df
    else:
        df = pd.read_sql(
            sql=f"SELECT * FROM {data_lookup_dict[f.transformation]}",
            con=sqlite3.connect(db_name),
        )
    return df


def correlation_report(df, numeric_features, db_name):
    # plt.subplots(nrows=1, ncols=len(numeric_features), sharey=True, figsize=(7, 4))
    fig = plt.figure(figsize=(16, 4))
    fig.tight_layout()
    fig.suptitle("Correlation of Numeric Features", fontsize=14, fontweight="bold")

    for i, j in enumerate(numeric_features):
        df = data_lookup(f=j, db_name=db_name, df=df)
        ax = fig.add_subplot(1, len(numeric_features), i + 1)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set(title=f"""{j.transformation} dataset""")
        sns.heatmap(df[j.features].corr(), annot=True)


def target_distribution_report(problem, df, target):
    _, ax = plt.subplots()
    ax.set_title("Target Distribution")
    if problem == "regression":
        sns.distplot(df[target])
    elif problem == "classification":
        sns.countplot(df[target])


def nan_report(df):
    name = "Percent of data encoded NAN"
    return pd.DataFrame(
        round((df.isna().sum() / df.shape[0]) * 100, 2), columns=[name],
    ).sort_values(by=name, ascending=False)


def skew_report(dataframe, threshold=5):
    highly_skewed = [
        i[0]
        for i in zip(dataframe.columns.values, abs(dataframe.skew(numeric_only=True)))
        if i[1] > threshold
    ]
    print(f"There are {len(highly_skewed)} highly skewed data columns.")
    if highly_skewed:
        print("Please check them for miscoded na's")
        print(highly_skewed)


def probability_plots(numeric_features, db_name, df):
    total_charts = len([i for i in numeric_features for j in i.features])
    fig = plt.figure(figsize=(12, total_charts * 4))
    fig.tight_layout()
    chart_count = 1
    for j in numeric_features:
        df = data_lookup(f=j, db_name=db_name, df=df)
        for k in j.features:
            ax1 = fig.add_subplot(total_charts, 2, chart_count)
            chart_count += 1
            probplot(df[k], plot=plt)
            plt.subplots_adjust(
                left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.35
            )
            ax1.set(title=f"Probability Plot:{k}:{j.transformation}".title())
            ax2 = fig.add_subplot(total_charts, 2, chart_count)
            chart_count += 1
            sns.distplot(df[k])
            ax2.set(title=f"Distribution Plot:{k}:{j.transformation}".title())


if __name__ == "__main__":
    db_name = "test.db"
    csv = "data/titanic/train.csv"
    problem = "classification"
    target = "Survived"
    create_notebook(db_name=db_name, csv=csv, problem=problem, target=target)
    run_notebook()
