from prefect import task
import prefect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
import numpy as np
from category_encoders.target_encoder import TargetEncoder
import re
from pyod.models.hbos import HBOS
import datetime
import sqlite3
import feather
from tinydb import TinyDB
from prefect import Flow, Parameter, unmapped


@task
def extract_train_valid_split(input_data, problem, target):
    if problem == "binary classification":
        train_data, valid_data = train_test_split(
            input_data, shuffle=True, stratify=input_data[target],
        )
    elif problem == "regression":
        train_data, valid_data = train_test_split(input_data, shuffle=True,)

    return train_data, valid_data


@task
def recreate_tinydb(name="db.json"):
    tiny_db = TinyDB(name)
    tiny_db.purge()
    return tiny_db


@task
def extract_nan_features(input_data):
    """a little complicated. map creates a %nan values and returns the feature if greater than the threshold.
        filter simply filters out the false values """
    f = input_data.columns.values
    len_df = len(input_data)
    nan_features = list(
        filter(
            lambda x: x is not False,
            map(
                lambda x: x if input_data[x].isna().sum() / len_df > 0.25 else False, f,
            ),
        )
    )
    return nan_features


@task
def extract_train_data(train_valid_split):
    return train_valid_split[0]


@task
def extract_valid_data(train_valid_split):
    return train_valid_split[1]


@task
def extract_problematic_features(input_data):
    f = input_data.columns.values
    problematic_features = []
    for i in f:
        if "Id" in i:
            problematic_features.append(i)
        elif "ID" in i:
            problematic_features.append(i)
    return problematic_features


@task
def extract_undefined_features(
    input_data, features, target, nan_features, problematic_features
):
    # import pdb

    # pdb.set_trace()
    if features == "infer":
        undefined_features = list(input_data.columns)
        if target in undefined_features:
            undefined_features.remove(target)
    for i in nan_features:
        undefined_features.remove(i)
    for i in problematic_features:
        undefined_features.remove(i)
    return undefined_features


@task
def extract_numeric_features(input_data, undefined_features):
    numeric_features = []
    l = undefined_features
    for i in l:
        if input_data[i].dtype in ["float64", "float", "int", "int64"]:
            if len(input_data[i].value_counts()) / len(input_data) < 0.1:
                pass
            else:
                numeric_features.append(i)
    return numeric_features


@task
def extract_categorical_features(
    input_data, undefined_features, threshold=10,
):
    categorical_features = []
    to_remove = []
    l = undefined_features
    for i in l:
        if len(input_data[i].value_counts()) / len(input_data[i]) < 0.10:
            categorical_features.append(i)
    return categorical_features


@task
def fit_transform_missing_indicator(input_data, undefined_features):
    indicator = MissingIndicator()
    x = indicator.fit_transform(input_data[undefined_features])
    print(undefined_features)
    print(indicator.features_)
    columns = [
        f"missing_{input_data[undefined_features].columns[ii]}"
        for ii in list(indicator.features_)
    ]
    print(columns)
    missing_indicator_df = pd.DataFrame(x, columns=columns)
    missing_indicator_df[columns].replace({True: 1, False: 0})
    input_data.merge(missing_indicator_df, left_index=True, right_index=True)
    return input_data


@task
def fit_numeric_imputer(train_data, numeric_features):
    numeric_imputer = SimpleImputer(strategy="median", copy=True)
    numeric_imputer.fit(train_data[numeric_features])
    return numeric_imputer


@task
def impute_numeric_df(numeric_imputer, data, numeric_features):
    x = numeric_imputer.transform(data[numeric_features])
    x_labels = [i + "imputed_" for i in numeric_features]
    imputed_numeric_df = pd.DataFrame(x, columns=x_labels)
    return imputed_numeric_df


@task
def fit_yeo_johnson_transformer(train_imputed_numeric_df):
    yeo_johnson_transformer = PowerTransformer(method="yeo-johnson", copy=True)
    yeo_johnson_transformer.fit(train_imputed_numeric_df)
    return yeo_johnson_transformer


@task
def transform_yeo_johnson_transformer(data, yeo_johnson_transformer):
    yj = yeo_johnson_transformer.transform(data)
    columns = data.columns.values
    columsn = [i + "_yj" for i in columns]
    yj = pd.DataFrame(yj, columns=columns)
    return yj


@task
def fit_categorical_imputer(train_data, categorical_features):
    categorical_imputer = SimpleImputer(strategy="most_frequent", copy=True)
    categorical_imputer.fit(train_data[categorical_features])
    return categorical_imputer


@task
def transform_categorical_data(data, categorical_features, categorical_imputer):
    x = categorical_imputer.transform(data[categorical_features])
    x_labels = [i + "_imputed" for i in categorical_features]
    imputed_categorical_df = pd.DataFrame(x, columns=x_labels)
    return imputed_categorical_df


@task
def fit_target_transformer(problem, target, train_data):
    if problem == "binary classification":
        return train_data[target]
    elif problem == "regression":
        target_transformer = PowerTransformer(method="yeo-johnson", copy=True)
        target_transformer.fit(np.array(train_data[target]).reshape(-1, 1))
        return target_transformer


@task
def transform_target(problem, target, data, target_transformer):
    if problem == "binary classification":
        return data[target]
    elif problem == "regression":
        target_array = target_transformer.transform(
            np.array(data[target]).reshape(-1, 1)
        )
        target_array = pd.DataFrame(target_array, columns=[target])
        return target_array


@task
def fit_target_encoder(train_imputed_categorical_df, train_transformed_target):
    te = TargetEncoder(cols=train_imputed_categorical_df.columns.values)
    te.fit(X=train_imputed_categorical_df, y=train_transformed_target)
    return te


@task
def target_encoder_transform(target_encoder, imputed_categorical_df):
    te = target_encoder.transform(imputed_categorical_df)
    columns = list(
        map(
            lambda x: re.sub(r"_imputed", "_target_encoded", x),
            list(imputed_categorical_df.columns.values),
        )
    )
    te = pd.DataFrame(data=te, columns=columns)
    return te


@task
def merge_transformed_data(
    target_encoded_df, yeo_johnson_df,
):
    transformed_data = target_encoded_df.merge(
        yeo_johnson_df, left_index=True, right_index=True
    ).replace(np.nan, 0)
    return transformed_data


@task
def fit_hbos_transformer(train_transformed_data):
    hbos = HBOS()
    hbos.fit(train_transformed_data)
    return hbos


@task
def hbos_transform(data, hbos_transformer):
    hbos_transformed = hbos_transformer.predict(data)
    hbos_transformed = pd.DataFrame(data=hbos_transformed, columns=["HBOS"])
    return hbos_transformed


@task
def merge_hbos_df(transformed_data, hbos_df):
    transformed_data.merge(hbos_df, left_index=True, right_index=True)
    return transformed_data


@task
def create_prediction_db(problem, target):
    day = datetime.datetime.now().day
    month = datetime.datetime.now().month
    year = datetime.datetime.now().year
    conn = sqlite3.connect(f"{year}-{month}-{day}/{problem}-{target}.db")
    conn.close()


@task
def fit_model(model, train_data, target, problem):
    try:
        return model.fit(X=train_data, y=target)
    except AttributeError:
        logger = prefect.context.get("logger")
        logger.warning(f"Warning: Inappropriate model for {problem}.")


@task
def debug(train_data, valid_data):
    t = set(train_data.columns.values)
    v = set(valid_data.columns.values)
    for ii in t:
        if ii not in v:
            logger = prefect.context.get("logger")
            logger.info(f"{ii} in train data but not valid data")


@task
def predict_model(model, valid_data):
    return model.predict(X=valid_data)


@task
def save_data(df, path):
    with open(path, "w+"):
        pass
    df.to_feather(path)
    return


@task
def fit_svd(df):
    svd = TruncatedSVD()
    svd.fit(df)
    return svd


@task
def svd_transform(svd, df, name, tiny_db):
    data = svd.transform(df).T
    x = [float(ii) for ii in data[0]]
    y = [float(ii) for ii in data[1]]
    tiny_db.insert({"chunk": f"svd-{name}", "x": x, "y": y})
    return svd.transform(df)


@task
def spectral_clustering(df):
    s = SpectralClustering()
    s.fit()


def upsert(fields, query, db):
    q_result = db.search(query)
    if len(q_result) == 0:
        db.insert()


with Flow("data_cleaning") as data_cleaning_flow:
    input_data = Parameter("input_data")
    problem, target, features = (
        Parameter("problem"),
        Parameter("target"),
        Parameter("features"),
    )
    tinydb = recreate_tinydb()
    nan_features = extract_nan_features(input_data)
    problematic_features = extract_problematic_features(input_data)
    undefined_features = extract_undefined_features(
        input_data, features, target, nan_features, problematic_features
    )
    input_data_with_missing = fit_transform_missing_indicator(
        input_data, undefined_features
    )

    train_valid_split = extract_train_valid_split(
        input_data=input_data_with_missing, problem=problem, target=target
    )
    train_data = extract_train_data(train_valid_split)
    valid_data = extract_valid_data(train_valid_split)
    numeric_features = extract_numeric_features(input_data, undefined_features)
    categorical_features = extract_categorical_features(input_data, undefined_features)

    # numeric columns work
    numeric_imputer = fit_numeric_imputer(train_data, numeric_features)
    imputed_train_numeric_df = impute_numeric_df(
        numeric_imputer, train_data, numeric_features
    )
    imputed_valid_numeric_df = impute_numeric_df(
        numeric_imputer, valid_data, numeric_features
    )

    yeo_johnson_transformer = fit_yeo_johnson_transformer(imputed_train_numeric_df)
    yeo_johnson_train_transformed = transform_yeo_johnson_transformer(
        imputed_train_numeric_df, yeo_johnson_transformer
    )
    yeo_johnson_valid_transformed = transform_yeo_johnson_transformer(
        imputed_valid_numeric_df, yeo_johnson_transformer
    )

    # categorical columns work
    categorical_imputer = fit_categorical_imputer(train_data, categorical_features)
    imputed_train_categorical_df = transform_categorical_data(
        train_data, categorical_features, categorical_imputer
    )
    imputed_valid_categorical_df = transform_categorical_data(
        valid_data, categorical_features, categorical_imputer
    )

    target_transformer = fit_target_transformer(problem, target, train_data)
    transformed_train_target = transform_target(
        problem, target, train_data, target_transformer
    )
    transformed_valid_target = transform_target(
        problem, target, valid_data, target_transformer
    )

    target_encoder_transformer = fit_target_encoder(
        imputed_train_categorical_df, transformed_train_target
    )
    target_encoded_train_df = target_encoder_transform(
        target_encoder_transformer, imputed_train_categorical_df
    )
    target_encoded_valid_df = target_encoder_transform(
        target_encoder_transformer, imputed_valid_categorical_df
    )

    # merge_data
    transformed_train_df = merge_transformed_data(
        target_encoded_train_df, yeo_johnson_train_transformed,
    )
    transformed_valid_df = merge_transformed_data(
        target_encoded_valid_df, yeo_johnson_valid_transformed,
    )

    # outlierness
    hbos_transformer = fit_hbos_transformer(transformed_train_df)
    hbos_transform_train_data = hbos_transform(transformed_train_df, hbos_transformer)
    hbos_transform_valid_data = hbos_transform(transformed_valid_df, hbos_transformer)

    # merge outlierness
    transformed_train_df = merge_hbos_df(
        transformed_train_df, hbos_transform_train_data
    )
    transformed_valid_df = merge_hbos_df(
        transformed_valid_df, hbos_transform_valid_data
    )
    save_data(
        transformed_train_df, "transformed_train.df",
    )
    save_data(
        transformed_valid_df, "transformed_valid.df",
    )

if __name__ == "__main__":
    pass
