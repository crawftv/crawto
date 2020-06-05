"""Collection of function for the data cleaning flow and the flow itself"""
from dataclasses import dataclass, field
import re
import sqlite3
from typing import Tuple, List, Union, Optional
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
def create_sql_data_tables(db: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE data_tables (data_tables text)")
        conn.execute(
            """CREATE TABLE features (
            category text  NOT NULL ,
            feature_list blob NOT NULL)"""
        )
    return
@task
def drop_target(input_data:pd.DataFrame,target:str) -> pd.DataFrame:
    df = input_data.drop(columns=[target],axis=1)
    return df
@task
def target_df(input_data:pd.DataFrame,target:str) ->pd.DataFrame:
    return pd.DataFrame(input_data[target])

@task
def extract_train_valid_split(
    input_data: pd.DataFrame, problem: str, target: str
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    if problem == "classification":
        train_data, valid_data = train_test_split(
            input_data, shuffle=True, stratify=input_data[target],
        )
        return train_data, valid_data
    elif problem == "regression":
        train_data, valid_data = train_test_split(input_data, shuffle=True,)

        return train_data, valid_data
    return None


@task
def extract_nan_features(
    input_data: pd.DataFrame, db_name: str, sql: None
) -> List[str]:
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
        conn.execute(query, ["NaN", cloudpickle.dumps(nan_features)])
    return nan_features


@task
def extract_problematic_features(
    input_data: pd.DataFrame, db_name: str, sql: None
) -> List[str]:
    #    """Extracts problematic features from a data"""
    problematic_features = []
    for i in input_data.columns.values:
        if "Id" in i:
            problematic_features.append(i)
        elif "ID" in i:
            problematic_features.append(i)

    with sqlite3.connect(db_name) as conn:
        query = "INSERT INTO features VALUES (?,?)"
        conn.execute(query, ["problematic", cloudpickle.dumps(problematic_features)])

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
    input_data: pd.DataFrame, undefined_features: List[str], db_name: str, sql: None
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
        conn.execute(query, ["numeric", cloudpickle.dumps(numeric_features)])

    return numeric_features


@task
def extract_categorical_features(
    input_data: pd.DataFrame, undefined_features: List[str], db_name: str, sql: None
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
        conn.execute(query, ("numeric", cloudpickle.dumps(categorical_features)))
    return categorical_features


@task
def fit_transform_missing_indicator(
    input_data: pd.DataFrame, db_name: str, sql: None
) -> pd.DataFrame:
    indicator = MissingIndicator()
    x = indicator.fit_transform(input_data)
    missing_features = [
        f"missing_{input_data.columns[ii]}" for ii in list(indicator.features_)
    ]
    missing_indicator_df = pd.DataFrame(x, columns=missing_features)
    missing_indicator_df[missing_features].replace({True: 1, False: 0})

    with sqlite3.connect(db_name) as conn:
        query = "INSERT INTO features VALUES (?,?)"
        conn.execute(query, ("missing", cloudpickle.dumps(missing_features)))
    output_data = input_data.merge(missing_indicator_df, left_index=True, right_index=True)
    return output_data


@task
def get_missing_dfs(train_valid_split: pd.DataFrame, db_name: str, sql:None) -> pd.DataFrame:
    with sqlite3.connect(db_name) as conn:
        result = conn.execute(
            """SELECT feature_list FROM features WHERE category = "missing" """
        ).fetchone()
    features = cloudpickle.loads(result[0])
    return train_valid_split[features].reset_index()


@task
def fit_hbos_transformer(input_data: pd.DataFrame):
    try:
        hbos = HBOS()
        hbos.fit(input_data)
        return hbos
    except:
        breakpoint()

@task
def hbos_transform(data: pd.DataFrame, hbos_transformer):
    hbos_transformed = hbos_transformer.predict(data)
    hbos_transformed = pd.DataFrame(data=hbos_transformed, columns=["HBOS"])
    return hbos_transformed


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
def fit_target_transformer(problem: str, target_df: pd.DataFrame):
    if problem == "classification":
        return pd.DataFrame(target_df)
    elif problem == "regression":
        # might comeback to this
        # target_transformer = PowerTransformer(method="yeo-johnson", copy=True)
        # target_transformer.fit(np.array(train_data[target]).reshape(-1, 1))
        # return target_transformer
        return FunctionTransformer(np.log1p).fit(target_df.values)

@task
def transform_target(
    problem: str,  target_df: pd.DataFrame, target_transformer
) -> pd.DataFrame:
    if problem == "classification":
        return target_df
    elif problem == "regression":
        target_array = target_transformer.transform(
            np.array(target_df).reshape(-1, 1)
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
    import prefect
    logger = prefect.context.get("logger")
    logger.info(f"{df1.shape, df2.shape}")
    df3 = df1.merge(df2, left_index=True, right_index=True,validate="one_to_one")
    logger.info(F"{df3.shape}")
    return df3


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
    df.to_sql(table_name, con=sqlite3.connect(db), if_exists="replace", index=True)


with Flow("data_cleaning") as data_cleaning_flow:
    input_data = Parameter("input_data")
    problem = Parameter("problem")
    target = Parameter("target")
    db_name = Parameter("db_name")

    sql = create_sql_data_tables(db_name)
    data_ex_target = drop_target(input_data,target=target)
    nan_features = extract_nan_features(data_ex_target, db_name=db_name, sql=sql)
    problematic_features = extract_problematic_features(
        data_ex_target, db_name=db_name, sql=sql
    )
    undefined_features = extract_undefined_features(
        data_ex_target, target, nan_features, problematic_features,
    )

    missing_full_df= fit_transform_missing_indicator(input_data, db_name=db_name, sql=sql,)
    train_valid_data = extract_train_valid_split(
        input_data= missing_full_df, problem=problem, target=target
    )

    numeric_features = extract_numeric_features(
        data_ex_target, undefined_features, db_name=db_name, sql=sql
    )
    categorical_features = extract_categorical_features(
        data_ex_target, undefined_features, db_name=db_name, sql=sql
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

    target_df = target_df.map(target = unmapped(target),input_data = train_valid_data)
    target_transformer = fit_target_transformer(problem=problem, target_df = target_df[0], )
    transformed_target = transform_target.map(
        unmapped(problem),
        target_df,
        unmapped(target_transformer),
    )

    target_encoder_transformer = fit_target_encoder(
        imputed_categorical_dfs[0], transformed_target[0]
    )
    target_encoded_dfs = target_encoder_transform.map(
        unmapped(target_encoder_transformer), imputed_categorical_dfs,
    )

    cat_num_dfs = merge_transformed_data.map(
        df1=imputed_categorical_dfs + target_encoded_dfs,
        df2=imputed_numeric_dfs + yeo_johnson_dfs,
    )

    df_to_sql.map(
        table_name=["transformed_train_target_df", "transformed_valid_target_df"],
        db=unmapped(db_name),
        df=transformed_target,
    )

    # outlierness
    hbos_transformer = fit_hbos_transformer(cat_num_dfs[2])
    hbos_transform_dfs = hbos_transform.map(cat_num_dfs[2:], unmapped(hbos_transformer))

    # merge outlierness
    cat_num_hbos = merge_transformed_data.map(
        df1=cat_num_dfs[2:], df2=hbos_transform_dfs
    )
    df_to_sql.map(
        table_name=["hbos1","hbos2"],
        db=unmapped(db_name),
        df = cat_num_hbos
    )
    missing_dfs = get_missing_dfs.map(train_valid_split=train_valid_data, db_name=unmapped(db_name),sql=unmapped(sql))
    df_to_sql.map(
        table_name=["missing_df1", "missing_df2"],
        db = unmapped(db_name),
        df = missing_dfs
    )
    cat_num_hbos_missing = merge_transformed_data.map(df1=cat_num_hbos, df2=missing_dfs)
    df_to_sql.map(
        table_name=[
            "imputed_train_df",
            "imputed_valid_df",
            "transformed_train_df",
            "transformed_valid_df",
        ],
        db=unmapped(db_name),
        df=cat_num_dfs[0:2] + cat_num_hbos_missing,
    )


def run_data_cleaning_flow(
    data_cleaning_flow,
    input_df: pd.DataFrame,
    problem: str,
    target: str,
    db_name: str = "crawto.db",
) -> None:
    executor = DaskExecutor()
    flow_state = data_cleaning_flow.run(
        input_data=input_df,
        problem=problem,
        target=target,
        db_name=db_name,
#        executor=executor,
    )
    data_cleaning_flow.visualize(flow_state=flow_state)
