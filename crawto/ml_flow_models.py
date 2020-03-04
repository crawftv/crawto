#!/usr/bin/env python3
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import(

    DecisionTreeRegressor,
    DecisionTreeClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    ElasticNet,
    Ridge,
    RidgeClassifier,
)
from baseline_model import (
    BaselineClassificationPrediction,
    BaselineRegressionPrediction
    )


def generate_baseline_prediction(problem,):
    if problem == "classification":
        return BaselineClassificationPrediction()
    #     classification_visualization(valid_data[target], y_pred,y_pred)
    elif problem == "regression":
        return BaselineRegressionPrediction()

def generate_regression_model(problem,):
    if problem == "classification":
        lr = LogisticRegression()
        return lr
    elif problem == "regression":
        lr = LinearRegression()
        return lr


def generate_decision_tree_model(problem,):
    if problem == "classification":
        dt = DecisionTreeClassifier(class_weight="balanced")
        return dt
    elif problem == "regression":
        dt = DecisionTreeRegressor()
        return dt


def generate_svm_model(problem):
    if problem == "classification":
        svm = LinearSVC()
        return svm


def generate_random_forest_model(problem):
    if problem == "classification":
        rf = RandomForestClassifier()
        return rf
    elif problem == "regression":
        rf = RandomForestRegressor()
        return rf


def generate_gradient_boosted_model(problem):
    if problem == "classification":
        gb = GradientBoostingClassifier()
        return gb
    elif problem == "regresssion":
        gb = GradientBoositingRegressor()
        return gb


def generate_ridge_model(problem):
    if problem == "classification":
        rm = RidgeClassfier()
        return rm
    elif problem == "regression":
        rm = Ridge()
        return rm


def generate_elastic_net(problem):
    if problem == "regression":
        en = ElasticNet()
        return en


def generate_all_models(problem):
    baseline = generate_baseline_prediction(problem)
    regression = generate_regression_model(problem)
    decision_tree = generate_decision_tree_model(problem)
    svm = generate_svm_model(problem)
    random_forest = generate_random_forest_model(problem)
    gbm = generate_gradient_boosted_model(problem)
    ridge = generate_ridge_model(problem)
    elasticnet = generate_elastic_net(problem)

    models = [regression, decision_tree, svm, random_forest, gbm, ridge, elasticnet]
    return models
