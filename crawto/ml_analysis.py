import json
import sqlite3
from dataclasses import asdict, dataclass, field
import crawto.classification_visualization as cv
from typing import Dict, List, Union, Tuple, Any
import numpy as np
import cloudpickle
import matplotlib.pyplot as plt
import pandas as pd
import papermill
import seaborn as sns
from scipy.stats import probplot
from torchnca import NCA
import torch
from scipy.stats import shapiro
import missingno
from IPython.display import HTML


@dataclass
class FeatureList:
    category: str
    feature_pickle: List = field(repr=False)

    @property
    def features(self):
        self._features = cloudpickle.loads(self.feature_pickle)
        return self._features

    @features.setter
    def features(self):
        return self._features


@dataclass
class Predictions:
    identifier: str
    scores: List = field(repr=False)
    predict_proba_pickle: Any
    dataset: str
    score: float

    @property
    def predictions(self):
        self._predictions = cloudpickle.loads(self.scores)
        return self._predictions

    @predictions.setter
    def predictions(self):
        return self._predictions

    @property
    def predict_proba(self):
        try:
            self._predict_proba = cloudpickle.loads(self.predict_proba_pickle)
        except TypeError:
            return None

    @predict_proba.setter
    def predict_proba(self):
        return self._predict_proba

    def visualization(self, transformed_data):
        cv.classification_visualization(
            transformed_data, self.predictions, self.predict_proba, self.identifier
        )

    @property
    def predictions(self):
        self._predictions = cloudpickle.loads(self.scores)
        return self._predictions
    @predictions.setter
    def predictions(self):
        return self._predictions
    @property
    def predict_proba(self):
        try:
            self._predict_proba = cloudpickle.loads(self.predict_proba_pickle)
        except TypeError:
           return None
    @predict_proba.setter
    def predict_proba(self):
        return self._predict_proba

    def visualization(self,transformed_data):
       cv.classification_visualization(transformed_data,self.predictions,self.predict_proba,self.identifier)

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
    import_cell.add("import missingno as msno")
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
    autoreload1_cell = asdict(Cell().add("%load_ext autoreload"))
    autoreload2_cell = asdict(Cell().add("%autoreload 2"))
    import_cell = asdict(
        create_import_cell(db_name=db_name, problem=problem, target=target)
    )
    load_df = asdict(
        Cell().add(
            f"untransformed_df, imputed_df,transformed_df,train_target_column, valid_target_column = ca.load_dfs(db_name)"
        )
    )
    df_list = asdict(
        Cell().add(
            """df_list = {"untransformed":untransformed_df,"imputed":imputed_df,"transformed":transformed_df}"""
        )
    )
    feature_list = asdict(create_feature_list_cell())
    # missingno
    na_report_cell = asdict(Cell().add("ca.nan_report(untransformed_df)"))
    missingno_matrix = asdict(Cell().add("msno.matrix(untransformed_df)"))
    missingno_bar = asdict(Cell().add("msno.bar(untransformed_df)"))
    missingno_heatmap = asdict(Cell().add("msno.heatmap(untransformed_df)"))
    missingno_dendrogram = asdict(Cell().add("msno.dendrogram(untransformed_df)"))

    skew_report_cell = asdict(Cell().add("ca.skew_report(untransformed_df)"))
    target_report_cell = asdict(
        Cell().add(
            "ca.target_distribution_report(problem=problem,df=untransformed_df,target=target)"
        )
    )
    # numeric feature analysis
    correlation_report_cell = asdict(
        Cell().add("ca.correlation_report(df_list,numeric_features,db_name)")
    )
    probability_plot_cell = asdict(
        Cell().add("ca.probability_plots(numeric_features,df_list)")
    )
    shapiro_distribution_cell = asdict(
        Cell().add("ca.distribution_r(df_list,numeric_features,target)")
    )
    # categorical feature analysis
    categorical_plot_cell = asdict(
        Cell().add(
            "ca.categorical_bar_plots(categorical_features=categoric_features,target=target,data=untransformed_df)"
        )
    )
    # dimensional reduction visualization
    svd_viz_cell = asdict(
        Cell()
        .add("from sklearn.decomposition import TruncatedSVD")
        .add(
            "ca.dimension_reduction_viz(transformed_df,train_target_column,target,problem,model=TruncatedSVD,title='SVD')"
        )
    )
    tsne_viz_cell = asdict(
        Cell()
        .add("from sklearn.manifold import TSNE")
        .add(
            "ca.dimension_reduction_viz(transformed_df,train_target_column,target,problem,model=TSNE,title='TSNE')"
        )
    )
    umap_viz_cell = asdict(
        Cell()
        .add("from umap import UMAP")
        .add(
            "ca.dimension_reduction_viz(transformed_df,train_target_column,target,problem,model=UMAP,title='UMAP')"
        )
    )
    nca_viz_cell = asdict(
        Cell().add("ca.nca_viz(transformed_df,train_target_column,target,problem)")
    )

    # model prediction visualizations

    model_viz_cell = asdict(Cell().add("ca.model_viz(db_name,valid_target_column)"))
    cells = [
        autoreload1_cell,
        autoreload2_cell,
        import_cell,
        load_df,
        df_list,
        feature_list,
        na_report_cell,
        missingno_matrix,
        missingno_bar,
        missingno_heatmap,
        missingno_dendrogram,
        skew_report_cell,
        target_report_cell,
        # numeric visualization
        correlation_report_cell,
        probability_plot_cell,
        shapiro_distribution_cell,
        # categorical visualization
        categorical_plot_cell,
        # matplotlib_charts,
        svd_viz_cell,
        tsne_viz_cell,
        umap_viz_cell,
        nca_viz_cell,
        # model prediction visualization
        model_viz_cell,
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

    target_valid_column = pd.read_sql(
        "SELECT * FROM transformed_valid_target_df", con=conn
    )
    return (
        untransformed_df,
        imputed_df,
        transformed_df,
        target_train_column,
        target_valid_column,
    )


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


def distribution_r(df_list, numeric_features: List[FeatureList], target: str):

    for index, value in enumerate(df_list):
        data = pd.DataFrame(
            [
                distribution_fit(df_list[value], feature)
                for feature in numeric_features.features
            ],
            index=numeric_features.features,
        )
        display(HTML(f"<h1>{value}</h1>"))
        display(data)


def distribution_fit(data, numeric_features):
    """
    x is a column_name
    """
    shapiro_values = shapiro(data[numeric_features])
    test_indication = True if shapiro_values[1] > 0.05 else False

    distribution_types = ["norm", "expon", "logistic", "gumbel"]
    # anderson_values = anderson(automl.data[numeric_column], dist=i)

    return {
        "Shapiro-Wilks_Test_Statistic": shapiro_values[0],
        "Shapiro-Wilks_p_Value": shapiro_values[1],
        "Normal distribution ?": test_indication
        # "Anderson_Darling_Test_Statistic_Normal": anderson_values[0][0],
    }


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


def dimension_reduction_viz(df, target_column, target, problem, model, title):
    fig = plt.figure(figsize=(12, 12))
    viz = model(n_components=2).fit_transform(df)
    viz_df = (
        pd.DataFrame(data=viz, columns=["X", "Y"])
        .merge(target_column, left_index=True, right_index=True)
        .merge(df["HBOS"], left_index=True, right_index=True)
    )
    if problem == "classification":
        ax1 = fig.add_subplot(2, 2, 1)
        sns.scatterplot(x="X", y="Y", hue=target, data=viz_df)
        ax1.set(title=f"{title} Vizualization of each classification")
        fig.add_subplot(2, 2, 2)
        ax2 = sns.scatterplot(x="X", y="Y", hue="HBOS", data=viz_df)
        ax2.set(title=f"{title} Vizualization of Outlierness")
    elif problem == "regression":
        fig = plt.figure(figsize=(12, 12))
        ax2 = sns.scatterplot(x="X", y="Y", hue="HBOS", data=viz_df)
        ax2.set(title=f"{title} Vizualization of Outlierness")


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
        ax1.set(title="NCA Vizualization of each classification")
        fig.add_subplot(2, 2, 2)
        ax2 = sns.scatterplot(x="X", y="Y", hue="HBOS", data=nca_df)
        ax2.set(title="NCA Vizualization of Outlierness")
    elif problem == "regression":
        fig = plt.figure(figsize=(12, 12))
        ax2 = sns.scatterplot(x="X", y="Y", hue="HBOS", data=nca_df)
        ax2.set(title="NCA Vizualization of Outlierness")



def model_viz(db_name, transformed_data):
    with sqlite3.connect(db_name) as conn:
        conn.row_factory = lambda c, r: Predictions(*r)
        query = """SELECT * FROM predictions"""
        rows = conn.execute(query).fetchall()
    for i in rows:
        i.visualization(transformed_data)


if __name__ == "__main__":
    DB_NAME = "test.db"
    CSV = "data/titanic/train.csv"
    PROBLEM = "classification"
    TARGET = "Survived"
    create_notebook(db_name=DB_NAME, csv=CSV, problem=PROBLEM, target=TARGET)
#    papermill.execute_notebook("crawto.ipynb", "crawto.ipynb")
