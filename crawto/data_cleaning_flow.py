"""Collection of function for the data cleaning flow and the flow itself"""
import re
import sqlite3
from typing import Tuple, List, Union
import cloudpickle
import numpy as np
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
from prefect import Flow, Parameter, task
from prefect.engine.executors import DaskExecutor
from pyod.models.hbos import HBOS
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PowerTransformer


@task
def extract_train_valid_split(
    input_data: pd.DataFrame, problem: str, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if problem == "classification":
        train_data, valid_data = train_test_split(
            input_data, shuffle=True, stratify=input_data[target],
        )
    elif problem == "regression":
        train_data, valid_data = train_test_split(input_data, shuffle=True,)

    return train_data, valid_data


@task
def extract_nan_features(input_data: pd.DataFrame) -> List[str]:
    #    """Adds a feature to a list if more than 25% of the values are nans """
    nan_features = input_data.columns.values
    len_df = len(input_data)
    return list(
        filter(
            lambda x: x is not False,
            map(
                lambda x: x if input_data[x].isna().sum() / len_df > 0.25 else False,
                nan_features,
            ),
        )
    )


@task
def extract_problematic_features(input_data: pd.DataFrame) -> List[str]:
    #    """Extracts problematic features from a data"""
    problematic_features = []
    for i in input_data.columns.values:
        if "Id" in i:
            problematic_features.append(i)
        elif "ID" in i:
            problematic_features.append(i)
    return problematic_features


@task
def extract_undefined_features(
    input_data: pd.DataFrame,
    target: str,
    nan_features: List[str],
    problematic_features: List[str],
) -> List[str]:

    undefined_features = list(input_data.columns.values)
    if target in undefined_features:
        undefined_features.remove(target)
    for i in nan_features:
        undefined_features.remove(i)
    for i in problematic_features:
        undefined_features.remove(i)
    return undefined_features


@task
def extract_numeric_features(
    input_data: pd.DataFrame, undefined_features: List[str]
) -> List[str]:
    l = undefined_features
    return [
        i
        for i in l
        if input_data[i].dtype in ["float64", "float", "int", "int64"]
        and len(input_data[i].value_counts()) / len(input_data) >= 0.01
    ]


@task
def extract_categorical_features(
    input_data: pd.DataFrame, undefined_features: List[str]
) -> List[str]:
    l = input_data.columns
    return [
        i
        for i in l
        if input_data[i].dtype in ["int64", "int", "object"]
        and len(input_data[i].value_counts()) / len(input_data) < 0.10
    ]


@task
def fit_transform_missing_indicator(
    input_data: pd.DataFrame, undefined_features: List[str]
) -> pd.DataFrame:
    indicator = MissingIndicator()
    x = indicator.fit_transform(input_data[undefined_features])
    columns = [
        f"missing_{input_data[undefined_features].columns[ii]}"
        for ii in list(indicator.features_)
    ]
    missing_indicator_df = pd.DataFrame(x, columns=columns)
    missing_indicator_df[columns].replace({True: 1, False: 0})
    input_data.merge(missing_indicator_df, left_index=True, right_index=True)
    return input_data


@task
def fit_hbos_transformer(input_data: pd.DataFrame):
    hbos = HBOS()
    hbos.fit(input_data)
    return hbos


@task
def hbos_transform(data: pd.DataFrame, hbos_transformer):
    hbos_transformed = hbos_transformer.predict(data)
    hbos_transformed = pd.DataFrame(data=hbos_transformed, columns=["HBOS"])
    return hbos_transformed


@task(name="merge_hbos_df")
def merge_hbos_df(transformed_data: pd.DataFrame, hbos_df: pd.DataFrame):
    transformed_data.merge(hbos_df, left_index=True, right_index=True)
    return transformed_data


@task
def extract_train_data(
    train_valid_split: Tuple[pd.DataFrame, pd.DataFrame]
) -> pd.DataFrame:
    return train_valid_split[0]


@task
def extract_valid_data(
    train_valid_split: Tuple[pd.DataFrame, pd.DataFrame]
) -> pd.DataFrame:
    return train_valid_split[1]


@task
def fit_numeric_imputer(
    train_data: pd.DataFrame, numeric_features: List[str]
) -> List[str]:
    numeric_imputer = SimpleImputer(strategy="median", copy=True)
    numeric_imputer.fit(train_data[numeric_features])
    return numeric_imputer


@task
def impute_numeric_df(
    numeric_imputer, data: pd.DataFrame, numeric_features: List[str]
) -> pd.DataFrame:
    imputed_df = numeric_imputer.transform(data[numeric_features])
    x_labels = [i for i in numeric_features]
    return pd.DataFrame(imputed_df, columns=x_labels)


@task
def fit_yeo_johnson_transformer(train_imputed_numeric_df: pd.DataFrame):
    yeo_johnson_transformer = PowerTransformer(method="yeo-johnson", copy=True)
    yeo_johnson_transformer.fit(train_imputed_numeric_df)
    return yeo_johnson_transformer


@task
def transform_yeo_johnson_transformer(data: pd.DataFrame, yeo_johnson_transformer):
    yjt = yeo_johnson_transformer.transform(data)
    columns = data.columns.values
    columns = [i for i in columns]
    return pd.DataFrame(yjt, columns=columns)


@task
def fit_categorical_imputer(train_data: pd.DataFrame, categorical_features: List[str]):
    categorical_imputer = SimpleImputer(strategy="most_frequent", copy=True)
    categorical_imputer.fit(train_data[categorical_features])
    return categorical_imputer


@task
def transform_categorical_data(
    data: pd.DataFrame, categorical_features: List[str], categorical_imputer
) -> pd.DataFrame:
    imputed = categorical_imputer.transform(data[categorical_features])
    x_labels = [i for i in categorical_features]
    return pd.DataFrame(imputed, columns=x_labels)


@task
def save_features(
    db_name,
    nan_features,
    problematic_features,
    numeric_features,
    categorical_features,
    imputed_train_numeric_df,
    yeo_johnson_train_transformed,
    target_encoded_train_df,
    imputed_train_categorical_df,
):
    nan = cloudpickle.dumps(list(nan_features))
    prob = cloudpickle.dumps(list(problematic_features))
    numeric = cloudpickle.dumps(list(numeric_features))
    categoric = cloudpickle.dumps(list(categorical_features))
    itn = cloudpickle.dumps(list(imputed_train_numeric_df.columns.values))
    itc = cloudpickle.dumps(list(imputed_train_categorical_df.columns.values))
    yjc = cloudpickle.dumps(list(yeo_johnson_train_transformed.columns.values))
    tec = cloudpickle.dumps(list(target_encoded_train_df.columns.values))
    execution_tuples = [
        ("nan", "un_transformed", nan),
        ("problematic", "untransformed", prob),
        ("numeric", "untransformed", numeric),
        ("categoric", "untransformed", categoric),
        ("numeric", "imputed", itn),
        ("categoric", "imputed", itc),
        ("numeric", "transformed", yjc),
        ("categoric", "transformed", tec),
    ]
    with sqlite3.connect(db_name) as conn:
        conn.execute(
            """CREATE TABLE features (
            category text  NOT NULL ,
            tranformation NOT NULL, 
            feature_list blob NOT NULL)"""
        )
        query = "INSERT INTO features VALUES(?,?,?)"
        conn.executemany(query, execution_tuples)


@task
def fit_target_transformer(problem: str, target: str, train_data: pd.DataFrame):
    if problem == "classification":
        return pd.DataFrame(train_data[target])
    elif problem == "regression":
        # might comeback to this
        # target_transformer = PowerTransformer(method="yeo-johnson", copy=True)
        # target_transformer.fit(np.array(train_data[target]).reshape(-1, 1))
        # return target_transformer
        target_transformer = FunctionTransformer(np.log1p)
        target_transformer.fit(train_data[target].values)
        return target_transformer


@task
def transform_target(
    problem: str, target: str, data: pd.DataFrame, target_transformer
) -> pd.DataFrame:
    if problem == "classification":
        return data[target]
    elif problem == "regression":
        target_array = target_transformer.transform(
            np.array(data[target]).reshape(-1, 1)
        )
        target_array = pd.DataFrame(target_array, columns=[target])
        return target_array


@task
def fit_target_encoder(
    train_imputed_categorical_df: pd.DataFrame, train_transformed_target: pd.DataFrame
):
    target_encoder = TargetEncoder(cols=train_imputed_categorical_df.columns.values)

    target_encoder.fit(X=train_imputed_categorical_df, y=train_transformed_target)
    return target_encoder


@task
def target_encoder_transform(target_encoder, imputed_categorical_df: pd.DataFrame):
    te = target_encoder.transform(imputed_categorical_df)
    columns = list(
        map(
            lambda x: re.sub(r"_imputed", "_target_encoded", x),
            list(imputed_categorical_df.columns.values),
        )
    )

    te = pd.DataFrame(data=te.values, columns=columns)
    return te


@task
def merge_transformed_data(
    target_encoded_df: pd.DataFrame, yeo_johnson_df: pd.DataFrame
) -> pd.DataFrame:
    return target_encoded_df.merge(yeo_johnson_df, left_index=True, right_index=True)


@task
def create_sql_data_tables(db: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE data_tables (data_tables text)")
    return


def np_to_sql_type(dtype: np.dtype) -> Union[str, None]:
    if pd.api.types.is_string_dtype(dtype):
        return "text"
    elif pd.api.types.is_float_dtype(dtype):
        return "real"
    elif pd.api.types.is_integer_dtype(dtype):
        return "int"
    else:
        return None


def df_to_sql_schema(table_name: str, df: pd.DataFrame) -> Tuple[str, List[str]]:
    if hasattr(df, "columns"):
        column_names = df.columns.values
        sql_types = list(map(np_to_sql_type, df.dtypes.values))
    else:
        column_names = [df.name]
        sql_types = list(map(np_to_sql_type, [df.dtype]))
    zz = list(zip(column_names, sql_types))
    pre_schema = [f""" [{i[0]}] {i[1]}""" for i in zz]
    schema = f"""({", ".join(pre_schema)})"""
    return schema, column_names


@task
def df_to_sql(table_name: str, db: str, df: pd.DataFrame) -> None:
    schema, _ = df_to_sql_schema(table_name, df)
    with sqlite3.connect(db) as conn:
        conn.execute(f"CREATE TABLE {table_name} {schema}")
        conn.execute("INSERT INTO data_tables VALUES (?)", (table_name,))
    df.to_sql(table_name, con=sqlite3.connect(db), if_exists="replace", index=False)


with Flow("data_cleaning") as data_cleaning_flow:
    input_data = Parameter("input_data")
    problem = Parameter("problem")
    target = Parameter("target")
    db_name = Parameter("db_name")

    create_sql_data_tables(db_name)
    nan_features = extract_nan_features(input_data)
    problematic_features = extract_problematic_features(input_data)
    undefined_features = extract_undefined_features(
        input_data, target, nan_features, problematic_features
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
    imputed_train_df = merge_transformed_data(
        imputed_train_categorical_df, imputed_train_numeric_df,
    )

    imputed_valid_df = merge_transformed_data(
        imputed_valid_categorical_df, imputed_valid_numeric_df,
    )

    transformed_train_df = merge_transformed_data(
        target_encoded_train_df, yeo_johnson_train_transformed,
    )

    transformed_valid_df = merge_transformed_data(
        target_encoded_valid_df, yeo_johnson_valid_transformed,
    )

    df_to_sql(
        table_name="transformed_train_target_df",
        db=db_name,
        df=transformed_train_target,
    )
    df_to_sql(
        table_name="transformed_valid_target_df",
        db=db_name,
        df=transformed_valid_target,
    )
    save_features(
        db_name,
        nan_features,
        problematic_features,
        numeric_features,
        categorical_features,
        imputed_train_numeric_df,
        yeo_johnson_train_transformed,
        target_encoded_train_df,
        imputed_train_categorical_df,
    )

    # outlierness
    hbos_transformer = fit_hbos_transformer(transformed_train_df)
    hbos_transform_train_data = hbos_transform(transformed_train_df, hbos_transformer)
    hbos_transform_valid_data = hbos_transform(transformed_valid_df, hbos_transformer)

    # merge outlierness
    # imputed_train_df = merge_hbos_df(imputed_train_df, hbos_transform_train_data)
    # imputed_valid_df = merge_hbos_df(imputed_valid_df, hbos_transform_valid_data)
    # transformed_train_df = merge_hbos_df(
    #    transformed_train_df, hbos_transform_train_data
    # )
    # transformed_valid_df = merge_hbos_df(
    #    transformed_valid_df, hbos_transform_valid_data
    # )
    df_to_sql(
        table_name="imputed_train_df",
        db=db_name,
        df=imputed_train_df,
        upstream_tasks=[merge_hbos_df(imputed_train_df, hbos_transform_train_data)],
    )
    df_to_sql(
        table_name="imputed_valid_df",
        db=db_name,
        df=imputed_valid_df,
        upstream_tasks=[merge_hbos_df(imputed_valid_df, hbos_transform_valid_data)],
    )
    df_to_sql(
        table_name="transformed_train_df",
        db=db_name,
        df=transformed_train_df,
        upstream_tasks=[merge_hbos_df(transformed_train_df, hbos_transform_train_data)],
    )
    df_to_sql(
        table_name="transformed_valid_df",
        db=db_name,
        df=transformed_valid_df,
        upstream_tasks=[merge_hbos_df(transformed_valid_df, hbos_transform_valid_data)],
    )


def run_data_cleaning_flow(
    data_cleaning_flow,
    input_df: pd.DataFrame,
    problem: str,
    target: str,
    db_name: str = "crawto.db",
) -> None:
    executor = DaskExecutor()
    data_cleaning_flow.run(
        input_data=input_df,
        problem=problem,
        target=target,
        db_name=db_name,
        executor=executor,
    )
