import pandas as pd
import numpy as np


def na_report(dataframe):
    print("NA's in the DataFrame")
    print(dataframe.isna().sum())


def skew_report(dataframe, threshold=5):
    highly_skewed = [
        i[0]
        for i in zip(dataframe.columns.values, abs(dataframe.skew(numeric_only=True)))
        if i[1] > threshold
    ]
    print(
        "There are %d highly skewed data columns. Please check them for miscoded na's"
        % len(highly_skewed)
    )
    print(highly_skewed)
