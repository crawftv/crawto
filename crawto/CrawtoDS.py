import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from math import ceil
import pandas as pd
from scipy.stats import shapiro, boxcox, yeojohnson
from scipy.stats import probplot
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from category_encoders.target_encoder import TargetEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_transformer

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
        for i in self.undefined_features:
            if len(self.input_data[i].value_counts()) / len(self.input_data[i]) < 0.10:
                categorical_features.append(i)
        return categorical_features

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
        x_labels = ["imputed" + i for i in self.numeric_features]
        imputed_numeric_df = pd.DataFrame(x, columns=x_labels)
        return imputed_numeric_df

    @property
    def imputed_categorical_df(self):
        x = self.categorical_imputer.transform(
            self.input_data[self.categorical_features]
        )
        x_labels = ["imputed" + i for i in self.categorical_features]
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
        return self.target_encoder.transform(
            X=self.imputed_categorical_df, y=self.input_data[self.target]
        )

    #     @property
    #     def nan_features(self):
    #         l = self.undefined_features
    #         for i in l:
    #             if self.input_data[i].isna().sum() / len(self.input_data[i]) > 0.25:
    #                 self.problematic_columns.append(i)
    #                 to_remove.append(i)
    #         for j in to_remove:
    #             self.undefined_features.remove(j)
    def __repr__(self):
        s = f"\ttarget: {self.target}\n\
        undefined features: {self.undefined_features}\n\
        nan features: {self.nan_features}\n\
        problematic features: {self.problematic_features}\n\
        numeric_features: {self.numeric_features}\n\
        categorical_features: {self.categorical_features}"
        return s
