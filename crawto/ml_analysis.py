import sqlite3
from dataclasses import dataclass, asdict, field
from typing import List, Dict
import json
import papermill
import cloudpickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def create_import_cell(db_name):
    import_cell = Cell()
    import_cell.add("import crawto.ml_analysis as ca")
    import_cell.add("import pandas as pd")
    import_cell.add(f"""db_name = "{db_name}" """)
    return import_cell


def create_feature_list_cell():
    cell = Cell()
    cell.source.append(
        "numeric_features, categoric_features = ca.get_feature_lists(db_name)"
    )
    return cell


def create_notebook(csv: str, db_name: str = "crawto.db"):
    import_cell = asdict(create_import_cell(db_name))
    load_df = asdict(Cell().add(f"df = pd.read_csv('{csv}')"))
    na_report = asdict(Cell().add("ca.na_report(df)"))
    correlation_report = asdict(
        Cell().add("ca.correlation_report(df,numeric_features,db_name)")
    )
    feature_list = asdict(create_feature_list_cell())

    nb = {
        "cells": [import_cell, load_df, na_report, feature_list, correlation_report],
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


def correlation_report(df, numeric_features, db_name):
    # plt.subplots(nrows=1, ncols=len(numeric_features), sharey=True, figsize=(7, 4))
    fig = plt.figure(figsize=(16, 4))
    fig.tight_layout()
    data_lookup_dict = {
        "imputed": "imputed_train_df",
        "transformed": "transformed_train_df",
        "untransformed": None,
    }
    for i, j in enumerate(numeric_features):
        if not data_lookup_dict[j.transformation]:
            df = df
        else:
            with sqlite3.connect(db_name) as conn:
                df = pd.read_sql(
                    sql=f"SELECT * FROM {data_lookup_dict[j.transformation]}",
                    con=sqlite3.connect(db_name),
                )
        fig.add_subplot(1, len(numeric_features), i + 1)
        sns.heatmap(df[j.features].corr())


def na_report(df):
    return df.isna().sum()


if __name__ == "__main__":
    db_name = "test.db"
    csv = "data/titanic/train.csv"
    create_notebook(db_name=db_name, csv=csv)
    run_notebook()
