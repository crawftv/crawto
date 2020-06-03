"""Collection of function for the data cleaning flow and the flow itself"""
from dataclasses import dataclass, field
import re
import sqlite3
from typing import Tuple, List, Union
import cloudpickle
import numpy as np
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
from prefect import Flow, Parameter, task, unmapped
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
def extract_nan_features(input_data: pd.DataFrame, db_name: str) -> List[str]:
    #    """Adds a feature to a list if more than 25% of the values are nans """
    nan_features = input_data.columns.values
    len_df = len(input_data)

    nan_features = list(
        filter(
            lambda x: x is not False,
            map(
                lambda x: x if input_data[x].isna().sum() / len_df > 0.25 else False,
                nan_features,
            ),
        )
    )
    with sqlite3.connect(db_name) as conn:
        query = "INSERT INTO features VALUES (?,?)"
        conn.executemany(query, ["NaN", cloudpickle.dumps(nan_features)])


@task
def extract_problematic_features(input_data: pd.DataFrame, db_name: str) -> List[str]:
    #    """Extracts problematic features from a data"""
    problematic_features = []
    for i in input_data.columns.values:
        if "Id" in i:
            problematic_features.append(i)
        elif "ID" in i:
            problematic_features.append(i)

    with sqlite3.connect(db_name) as conn:
        query = "INSERT INTO features VALUES (?,?)"
        conn.executemany(
            query, ["problematic", cloudpickle.dumps(problematic_features)]
        )

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
    input_data: pd.DataFrame, undefined_features: List[str], db_name: str
) -> List[str]:
    l = undefined_features
    numeric_features = [
        i
        for i in l
        if input_data[i].dtype in ["float64", "float", "int", "int64"]
        and len(input_data[i].value_counts()) / len(input_data) >= 0.01
    ]
    with sqlite3.connect(db_name) as conn:
        query = "INSERT INTO features VALUES (?,?)"
        conn.executemany(query, ["numeric", cloudpickle.dumps(numeric_features)])

    return numeric_features


@task
def extract_categorical_features(
    input_data: pd.DataFrame, undefined_features: List[str], db_name: str
) -> List[str]:
    l = input_data.columns
    categorical_features = [
        i
        for i in l
        if input_data[i].dtype in ["int64", "int", "object"]
        and len(input_data[i].value_counts()) / len(input_data) < 0.10
    ]
    with sqlite3.connect(db_name) as conn:
        query = "INSERT INTO features VALUES (?,?)"
        conn.executemany(query, ("numeric", cloudpickle.dumps(numeric_features)))
    return categorical_features


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
def merge_transformed_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # breakpoint()
    return df1.merge(df2, left_index=True, right_index=True)


@task
def create_sql_data_tables(db: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE data_tables (data_tables text)")
        conn.execute(
            """CREATE TABLE features (
            category text  NOT NULL ,
            feature_list blob NOT NULL)"""
        )
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
    nan_features = extract_nan_features(input_data, db_name=db_name)
    problematic_features = extract_problematic_features(input_data, db_name=db_name)
    undefined_features = extract_undefined_features(
        input_data, target, nan_features, problematic_features
    )
    input_data_with_missing = fit_transform_missing_indicator(
        input_data, undefined_features
    )

    train_valid_data = extract_train_valid_split(
        input_data=input_data_with_missing, problem=problem, target=target
    )

    numeric_features = extract_numeric_features(
        input_data, undefined_features, db_name=db_name
    )
    categorical_features = extract_categorical_features(
        input_data, undefined_features, db_name=db_name
    )

    # numeric columns work
    numeric_imputer = fit_numeric_imputer(train_valid_data[0], numeric_features)
    imputed_numeric_dfs = impute_numeric_df.map(
        unmapped(numeric_imputer), train_valid_data, unmapped(numeric_features)
    )

    yeo_johnson_transformer = fit_yeo_johnson_transformer(imputed_numeric_dfs[0])
    yeo_johnson_dfs = transform_yeo_johnson_transformer.map(
        imputed_numeric_dfs, unmapped(yeo_johnson_transformer)
    )

    # categorical columns work
    categorical_imputer = fit_categorical_imputer(
        train_valid_data[0], categorical_features
    )
    imputed_categorical_dfs = transform_categorical_data.map(
        train_valid_data, unmapped(categorical_features), unmapped(categorical_imputer)
    )

    target_transformer = fit_target_transformer(problem, target, train_valid_data[0])
    transformed_target = transform_target.map(
        unmapped(problem),
        unmapped(target),
        train_valid_data,
        unmapped(target_transformer),
    )

    target_encoder_transformer = fit_target_encoder(
        imputed_categorical_dfs[0], transformed_target[0]
    )
    target_encoded_dfs = target_encoder_transform.map(
        unmapped(target_encoder_transformer), imputed_categorical_dfs,
    )

    transformed_df_map = merge_transformed_data.map(
        df1=imputed_categorical_dfs + target_encoded_dfs,
        df2=imputed_numeric_dfs + yeo_johnson_dfs,
    )

    df_to_sql.map(
        table_name=["transformed_train_target_df", "transformed_valid_target_df",],
        db=unmapped(db_name),
        df=transformed_target,
    )

    # outlierness
    hbos_transformer = fit_hbos_transformer(transformed_df_map[3])
    hbos_transform_dfs = hbos_transform.map(
        transformed_df_map[3:4], unmapped(hbos_transformer)
    )

    # merge outlierness
    transformed_train_df = merge_hbos_df.map(
        transformed_df_map[3:4], hbos_transform_dfs
    )
    df_to_sql.map(
        table_name=[
            "imputed_train_df",
            "imputed_valid_df",
            "transformed_train_df",
            "transformed_valid_df",
        ],
        db=unmapped(db_name),
        df=transformed_df_map,
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
