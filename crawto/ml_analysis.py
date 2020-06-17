import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Union, Tuple
import numpy as np
import cloudpickle
import matplotlib.pyplot as plt
import pandas as pd
import papermill
import seaborn as sns
from scipy.stats import probplot
from sklearn.manifold import TSNE
from umap import UMAP
from torchnca import NCA
import torch


@dataclass
class FeatureList:
    category: str
    feature_pickle: List = field(default_factory=list, repr=False)

    @property
    def features(self):
        return cloudpickle.loads(self.feature_pickle)


@dataclass
class Cell:
    cell_type: str = "code"
    execution_count: Union[int, None] = None
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


def create_import_cell(db_name: str, problem: str, target: str) -> Cell:
    import_cell = Cell()
    import_cell.add("import crawto.ml_analysis as ca")
    import_cell.add("import pandas as pd")
    import_cell.add(f"""db_name = "{db_name}" """)
    import_cell.add(f"""problem = "{problem}" """)
    import_cell.add(f"""target = "{target}" """)
    return import_cell


def create_feature_list_cell() -> Cell:
    cell = Cell()
    cell.source.append(
        "numeric_features, categoric_features = ca.get_feature_lists(db_name)"
    )
    return cell


def create_notebook(csv: str, problem: str, target: str, db_name: str) -> None:
    import_cell = asdict(
        create_import_cell(db_name=db_name, problem=problem, target=target)
    )
    load_df = asdict(
        Cell().add(
            f"untransformed_df, imputed_df,transformed_df,target_column = ca.load_dfs(db_name)"
        )
    )
    df_list = asdict(
        Cell().add(
            """df_list = {"untransformed":untransformed_df,"imputed":imputed_df,"transformed":transformed_df}"""
        )
    )
    na_report_cell = asdict(Cell().add("ca.nan_report(untransformed_df)"))
    skew_report_cell = asdict(Cell().add("ca.skew_report(untransformed_df)"))
    feature_list = asdict(create_feature_list_cell())
    correlation_report_cell = asdict(
        Cell().add("ca.correlation_report(df_list,numeric_features,db_name)")
    )
    target_report_cell = asdict(
        Cell().add(
            "ca.target_distribution_report(problem=problem,df=untransformed_df,target=target)"
        )
    )
    probability_plot_cell = asdict(
        Cell().add("ca.probability_plots(numeric_features,df_list)")
    )
    categorical_plot_cell = asdict(
        Cell().add(
            "ca.categorical_bar_plots(categorical_features=categoric_features,target=target,data=untransformed_df)"
        )
    )
    tsne_viz_cell = asdict(
        Cell().add("ca.tsne_viz(transformed_df,target_column,target,problem)")
    )
    umap_viz_cell = asdict(
        Cell().add("ca.umap_viz(transformed_df,target_column,target,problem)")
    )
    nca_viz_cell = asdict(
        Cell().add("ca.nca_viz(transformed_df,target_column,target,problem)")
    )
    categorical_plot_cell = asdict(Cell().add("ca.categorical_bar_plots(categorical_features=categorical_features,target=target,data=df)"))
    cells = [
        import_cell,
        load_df,
        df_list,
        na_report_cell,
        skew_report_cell,
        feature_list,
        correlation_report_cell,
        target_report_cell,
        probability_plot_cell,
        categorical_plot_cell,
        matplotlib_charts
        tsne_viz_cell,
        umap_viz_cell,
        nca_viz_cell,

    ]
    notebook = {
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
    with open("crawto.ipynb", "w") as filename:
        json.dump(notebook, filename)


def run_notebook() -> None:
    papermill.execute_notebook("crawto.ipynb", "crawto.ipynb")


# functions


def get_feature_lists(db_name: str) -> Tuple[List[str], List[str]]:
    with sqlite3.connect(db_name) as conn:
        conn.row_factory = lambda c, r: FeatureList(*r)
        numeric_features = conn.execute(
            "SELECT * FROM features where category = 'numeric'"
        ).fetchone()
        categoric_features = conn.execute(
            "SELECT * FROM features where category = 'categoric'"
        ).fetchone()
    return numeric_features, categoric_features


def load_dfs(db_name):
    conn = sqlite3.connect(db_name)
    untransformed_df = pd.read_sql("SELECT * FROM untransformed_train_df", con=conn)
    imputed_df = pd.read_sql("SELECT * FROM imputed_train_df", con=conn)
    transformed_df = pd.read_sql("SELECT * FROM transformed_train_df", con=conn)
    target_train_column = pd.read_sql(
        "SELECT * FROM transformed_train_target_df", con=conn
    )
    return untransformed_df, imputed_df, transformed_df, target_train_column


def correlation_report(
    df_list: List[pd.DataFrame], numeric_features: FeatureList, db_name: str
) -> None:
    fig = plt.figure(figsize=(16, 4))
    fig.tight_layout()
    fig.suptitle("Correlation of Numeric Features", fontsize=14, fontweight="bold")
    for i, transformation in enumerate(df_list):
        df = df_list[transformation]
        ax = fig.add_subplot(1, len(df_list), i + 1)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set(title=f"""{transformation} dataset""")
        sns.heatmap(df[numeric_features.features].corr(), annot=True)


def target_distribution_report(problem: str, df: pd.DataFrame, target: str) -> None:
    _, ax = plt.subplots()
    ax.set_title("Target Distribution")
    if problem == "regression":
        sns.distplot(df[target])
    elif problem == "classification":
        sns.countplot(df[target])


def nan_report(df: pd.DataFrame) -> pd.DataFrame:
    name = "Percent of data encoded NAN"
    return pd.DataFrame(
        round((df.isna().sum() / df.shape[0]) * 100, 2), columns=[name],
    ).sort_values(by=name, ascending=False)


def skew_report(dataframe: pd.DataFrame, threshold: int = 5) -> None:
    highly_skewed = [
        i[0]
        for i in zip(dataframe.columns.values, abs(dataframe.skew(numeric_only=True)))
        if i[1] > threshold
    ]
    print(f"There are {len(highly_skewed)} highly skewed data columns.")
    if highly_skewed:
        print("Please check them for miscoded na's")
        print(highly_skewed)


def probability_plots(numeric_features: List[FeatureList], df_list) -> None:
    total_charts = len([i for i in numeric_features.features]) * len(df_list)
    fig = plt.figure(figsize=(16, total_charts * 4))
    chart_count = 1
    for k in numeric_features.features:
        for transformation in df_list:
            df = df_list[transformation]

            ax1 = fig.add_subplot(total_charts, 3, chart_count)
            chart_count += 1
            probplot(df[k], plot=plt)
            ax1.set(title=f"Probability Plot:{k}:{transformation}".title())

            ax2 = fig.add_subplot(total_charts, 3, chart_count)
            chart_count += 1
            sns.distplot(df[k])
            ax2.set(title=f"Distribution Plot:{k}:{transformation}".title())

            ax3 = fig.add_subplot(total_charts, 3, chart_count)
            chart_count += 1
            ax3.set(title=f"Box Plot:{k}:{transformation}".title())
            sns.boxplot(data=df[k], orient="h")
    fig.tight_layout()


def categorical_bar_plots(categorical_features, target, data):
    categorical_features = categorical_features.features
    total_features = len(categorical_features)
    fig = plt.figure(figsize=(11, total_features * 4))
    chart_count = 1
    for i in range(total_features):
        ax1 = fig.add_subplot(total_features, 2, chart_count)
        sns.barplot(x=categorical_features[i - 0], y=target, data=data)
        ax1.set(title=f"% of each label that are {target}".title())
        chart_count += 1
        ax2 = fig.add_subplot(total_features, 2, chart_count)
        sns.countplot(x=categorical_features[i - 0], data=data)
        ax2.set(title="Count of each label".title())
        chart_count += 1
    fig.tight_layout()


def tsne_viz(df, target_column, target, problem):
    fig = plt.figure(figsize=(12, 12))
    tsne = TSNE(n_components=2).fit_transform(df)
    tsne_df = (
        pd.DataFrame(data=tsne, columns=["X", "Y"])
        .merge(target_column, left_index=True, right_index=True)
        .merge(df["HBOS"], left_index=True, right_index=True)
    )
    if problem == "classification":
        ax1 = fig.add_subplot(2, 2, 1)
        sns.scatterplot(x="X", y="Y", hue=target, data=tsne_df)
        ax1.set(title="TSNE Vizualization of each classification")
        fig.add_subplot(2, 2, 2)
        ax2 = sns.scatterplot(x="X", y="Y", hue="HBOS", data=tsne_df)
        ax2.set(title="TSNE Vizualization of Outlierness")
    elif problem == "regression":
        fig = plt.figure(figsize=(12, 12))
        ax2 = sns.scatterplot(x="X", y="Y", hue="HBOS", data=tsne_df)
        ax2.set(title="TSNE Vizualization of Outlierness")


def umap_viz(df, target_column, target, problem):
    fig = plt.figure(figsize=(12, 12))
    umap_df = UMAP().fit_transform(df)
    umap_df = (
        pd.DataFrame(data=umap_df, columns=["X", "Y"])
        .merge(target_column, left_index=True, right_index=True)
        .merge(df["HBOS"], left_index=True, right_index=True)
    )
    if problem == "classification":
        ax1 = fig.add_subplot(2, 2, 1)
        sns.scatterplot(x="X", y="Y", hue=target, data=umap_df)
        ax1.set(title="UMAP Vizualization of each classification")
        fig.add_subplot(2, 2, 2)
        ax2 = sns.scatterplot(x="X", y="Y", hue="HBOS", data=umap_df)
        ax2.set(title="UMAP Vizualization of Outlierness")
    elif problem == "regression":
        ax2 = sns.scatterplot(x="X", y="Y", hue="HBOS", data=umap_df)
        ax2.set(title="UMAP Vizualization of Outlierness")


def nca_viz(df, target_column, target, problem):
    df1 = df.copy()
    df1 = df1.values.astype(np.float32)
    target_column = target_column.values.astype(np.float32)
    nca = NCA(dim=2, init="identity")
    nca.train(
        torch.tensor(df1), torch.tensor(target_column), batch_size=32, normalize=True
    )
    new_df = nca(torch.tensor(df1)).detach().numpy()
    nca_df = (
        pd.DataFrame(data=new_df, columns=["X", "Y"])
        .merge(
            pd.DataFrame(data=target_column, columns=[target]),
            left_index=True,
            right_index=True,
        )
        .merge(df["HBOS"], left_index=True, right_index=True)
    )
    if problem == "classification":
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        sns.scatterplot(x="X", y="Y", hue=target, data=nca_df)
        ax1.set(title="UMAP Vizualization of each classification")
        fig.add_subplot(2, 2, 2)
        ax2 = sns.scatterplot(x="X", y="Y", hue="HBOS", data=nca_df)
        ax2.set(title="UMAP Vizualization of Outlierness")
    elif problem == "regression":
        fig = plt.figure(figsize=(12, 12))
        ax2 = sns.scatterplot(x="X", y="Y", hue="HBOS", data=nca_df)
        ax2.set(title="UMAP Vizualization of Outlierness")


def categorical_bar_plots(categorical_features,target,data):
    fig = plt.figure(figsize=(11, len(categorical_features) * 4))
    fig.tight_layout()
    chart_count = 0
    for i in range(0, len(categorical_features) + 1):
        fig.add_subplot(len(categorical_features)), 1, chart_count)
        sns.barplot(x=categorical_features[i - 0], y=target, data=data)
        chart_count += 1
        fig.add_subplot(len(categorical_features), 1, chart_count)
        sns.countplot(x=categorical_features[i - 0], data=data)
        chart_count += 1

if __name__ == "__main__":
    DB_NAME = "test.db"
    CSV = "data/titanic/train.csv"
    PROBLEM = "classification"
    TARGET = "Survived"
    create_notebook(db_name=DB_NAME, csv=CSV, problem=PROBLEM, target=TARGET)
#    run_notebook()
