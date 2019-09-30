import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from math import ceil
import pandas as pd
from sklearn.metrics import classification_report
from scipy.stats import shapiro, boxcox, yeojohnson
from scipy.stats import probplot
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from category_encoders.target_encoder import TargetEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_transformer
from crawto.classification_visualization import classification_visualization
from sklearn.linear_model import LogisticRegression

sns.set_palette("colorblind")


class CrawtoDS:
    def __init__(self, data, target, features="infer", problem="infer"):
        self.input_data = data
        self.target = target
        self.features = features
        self.problem = problem

    @property
    def nan_features(self):
        """a little complicated. map creates a %nan values and returns the feature if greater than the threshold.
        filter simply filters out the false values """
        f = self.input_data.columns.values
        len_df = len(self.input_data)
        nan_features = list(
            filter(
                lambda x: x is not False,
                map(
                    lambda x: x
                    if self.input_data[x].isna().sum() / len_df > 0.25
                    else False,
                    f,
                ),
            )
        )
        return nan_features

    @property
    def problematic_features(self):
        f = self.input_data.columns.values
        problematic_features = []
        for i in f:
            if "Id" in i:
                problematic_features.append(i)
            elif "ID" in i:
                problematic_features.append(i)
        return problematic_features

    @property
    def undefined_features(self):
        if self.features == "infer":
            undefined_features = list(self.input_data.columns)
            undefined_features.remove(self.target)
        for i in self.nan_features:
            undefined_features.remove(i)
        for i in self.problematic_features:
            undefined_features.remove(i)
        return undefined_features

    @property
    def numeric_features(self):
        numeric_features = []
        l = self.undefined_features
        for i in l:
            if self.input_data[i].dtype in ["float64", "float", "int", "int64"]:
                numeric_features.append(i)
        return numeric_features

    @property
    def categorical_features(self, threshold=10):
        self.undefined_features
        categorical_features = []
        to_remove = []
        l = self.undefined_features
        for i in l:
            if len(self.input_data[i].value_counts()) / len(self.input_data[i]) < 0.10:
                categorical_features.append(i)
        return categorical_features

    #     @categorical_features.setter
    #     def categorical_features(self,new_categorical_features_list):
    #         #self.categorical_features = new_categorical_features_list
    #         #self.
    #         return new_categorical_features_list

    @property
    def missing_indicator_df(self):
        x = self.indicator.transform(self.input_data[self.undefined_features])
        x_labels = ["missing_" + i for i in self.undefined_features]
        missing_indicator_df = pd.DataFrame(x, columns=x_labels)
        return missing_indicator_df

    @property
    def indicator(self):
        indicator = MissingIndicator(features="all")
        indicator.fit(self.input_data[self.undefined_features])
        return indicator

    @property
    def numeric_imputer(self):
        numeric_imputer = SimpleImputer(strategy="median", copy=True)
        numeric_imputer.fit(self.input_data[self.numeric_features])
        return numeric_imputer

    @property
    def categorical_imputer(self):
        categorical_imputer = SimpleImputer(strategy="most_frequent", copy=True)
        categorical_imputer.fit(self.input_data[self.categorical_features])
        return categorical_imputer

    @property
    def yeo_johnson_transformer(self):
        yeo_johnson_transformer = PowerTransformer(method="yeo-johnson", copy=True)
        yeo_johnson_transformer.fit(self.imputed_numeric_df)
        return yeo_johnson

    #         self.labelencoder = LabelEncoder()
    #         self.labelencoder.fit(self.imputed_categorical_df)

    @property
    def target_encoder(self):
        te = TargetEncoder(cols=self.imputed_categorical_df.columns.values)
        te.fit(X=self.imputed_categorical_df, y=self.input_data[self.target])
        return te

    @property
    def imputed_numeric_df(self):
        x = self.numeric_imputer.transform(self.input_data[self.numeric_features])
        x_labels = [i + "_imputed" for i in self.numeric_features]
        imputed_numeric_df = pd.DataFrame(x, columns=x_labels)
        return imputed_numeric_df

    @property
    def imputed_categorical_df(self):
        x = self.categorical_imputer.transform(
            self.input_data[self.categorical_features]
        )
        x_labels = [i + "_imputed" for i in self.categorical_features]
        imputed_categorical_df = pd.DataFrame(x, columns=x_labels)
        return imputed_categorical_df

    @property
    def yeojohnson_df(self):
        return self.yeojohnson_transformer.transform(self.imputed_numeric_df)

    #     @property
    #     def labelencoded_df(self):
    #         return self.labelencoder.transform(self.imputed_categorical_df)

    @property
    def target_encoded_categorical_df(self):
        te = self.target_encoder.transform(self.imputed_categorical_df)
        columns = list(
            map(
                lambda x: re.sub(r"_imputed", "_target_encoded", x),
                list(self.imputed_categorical_df.columns.values),
            )
        )
        te = pd.DataFrame(data=te)
        te.columns = columns
        return te

    def correlation_report(self):
        sns.heatmap(self.input_data[self.numeric_features].corr())

    def target_distribution_report(self):
        if self.problem == "regression":
            return sns.distplot(self.input_data[self.target])
        elif self.problem == "classification":
            return sns.countplot(self.input_data[self.target])

    def numeric_columns_distribution_report(self):
        return self.distribution_r()

    def distribution_r(self):
        display(
            pandas.DataFrame(
                [
                    self.distribution_fit(self.input_data, i)
                    for i in self.numeric_features + [self.target]
                ],
                index=self.numeric_features + [self.target],
            )
        )

    def distribution_fit(self, data, numeric_features):
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

    def nan_report(self):
        return display(
            pandas.DataFrame(
                round(
                    (self.input_data.isna().sum() / self.input_data.shape[0]) * 100, 2
                ),
                columns=["Percent of data encoded NAN"],
            )
        )

    def correlation_report(self, threshold=0.95):
        corr_matrix = self.input_data[[self.target] + self.numeric_features].corr()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )
        highly_correlated_features = [
            column for column in upper.columns if any(upper[column] > threshold)
        ]
        sns.heatmap(corr_matrix)
        if len(highly_correlated_features) > 0:
            return f"Highly Correlated features are {highly_correlated_features}"
        else:
            return "No Features are correlated above the threshold"

    def probability_plots(self):
        c = list(self.imputed_numeric_df.columns.values)
        c.sort()
        c.remove(self.target)
        fig = plt.figure(figsize=(12, len(c) * 4))
        fig.tight_layout()
        chart_count = 1
        for i in range(1, (len(c) + 1), 1):
            fig.add_subplot(len(c), 2, chart_count)
            chart_count += 1
            probplot(self.imputed_numeric_df[c[i - 1]], plot=plt)
            plt.subplots_adjust(
                left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.35
            )
            plt.title(c[i - 1] + " Probability Plot")
            fig.add_subplot(len(c), 2, chart_count)
            chart_count += 1
            sns.distplot(self.imputed_numeric_df[c[i - 1]])
            plt.show()

    def categorical_bar_plots(self):
        data = self.imputed_categorical_df.merge(
            self.input_data[self.target], left_index=True, right_index=True, sort=True
        )
        c = list(data.columns.values)
        c.sort()
        fig = plt.figure(figsize=(12, len(c) * 4))
        fig.tight_layout()
        chart_count = 1
        for i in range(1, len(c) + 1):
            fig.add_subplot(len(c), 2, chart_count)
            sns.barplot(x=c[i - 1], y=self.target, data=data)
            chart_count += 1
            fig.add_subplot(len(c), 2, chart_count)
            sns.countplot(x=c[i - 1], data=data)
            chart_count += 1

    def baseline_prediction(self):
        if self.problem == "classification":
            y_pred = np.dot(
                np.ones_like(self.input_data[self.target]).reshape(-1, 1),
                np.array(self.input_data[self.target].mode()).reshape(-1, 1),
            )
            classification_visualization(self.input_data[self.target], y_pred)
        if self.problem == "regression":
            pass

    def naive_regression(self):
        if self.problem == "classification":
            lr = LogisticRegression(penalty="none", c=0.0)
            naive_data = self.imputed_numeric_df.merge(
                self.imputed_categorical_df, left_index=True, right_index=True
            ).merge(self.missing_indicator_df, left_index=True, right_index=True)
            lr.fit(naive_data, self.input_data[self.target])

    def __repr__(self):
        s = f"\ttarget: {self.target}\n\
        undefined features: {self.undefined_features}\n\
        nan features: {self.nan_features}\n\
        problematic features: {self.problematic_features}\n\
        numeric_features: {self.numeric_features}\n\
        categorical_features: {self.categorical_features}"
        return s
