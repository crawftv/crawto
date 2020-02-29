#!/usr/bin/env python3
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    ElasticNet,
    Ridge,
    RidgeClassifier,
)


def generate_regression_model(problem,):
    if problem == "binary classification":
        lr = LogisticRegression()
        return lr
    elif problem == "regression":
        lr = LinearRegression()
        return lr


def generate_decision_tree_model(problem,):
    if problem == "binary classification":
        dt = DecisionTreeClassifier(class_weight="balanced")
        return dt
    elif problem == "regression":
        dt = DecisionTreeRegressor()
        return dt


def generate_svm_model(problem):
    if problem == "binary classification":
        svm = LinearSVC()
        return svc


def generate_random_forest_model(problem):
    if problem == "binary classification":
        rf = RandomForestClassifier()
        return rf
    elif problem == "regression":
        rf = RandomForestRegressor()
        return rf


def generate_gradient_boosted_model(problem):
    if problem == "binary classification":
        gb = GradientBoostingClassifier()
        return gb
    elif problem == "regresssion":
        gb = GradientBoositingRegressor()
        return gb


def generate_ridge_model(problem):
    if problem == "binary classification":
        rm = RidgeClassfier()
        return rm
    elif problem == "regression":
        rm = Ridge()
        return rm


def generate_elastic_net(problem):
    if problem == "regression":
        en = ElasticNet()
        return en
