#!/usr/bin/env python3
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, LogisticRegression


def generate_all_base_models(problem):
    [
        generate_regression_model(problem),
        generate_decicion_tree_model(problem)
    ]
def generate_regression_model(problem,*args,**kwargs):
    if problem == "binary classification":
        lr = LogisticRegression(*args,**kwargs)
        return lr
    elif problem == "regression":
        lr = LinearRegression(*args,**kwargs)
        return lr

def generate_decision_tree_model(problem,*args,**kwargs):
    if problem == "binary classification":
        dt = DecisionTreeClassifier(class_weight="balanced")
        return dt
    elif problem == "regression":
        dt = DecisionTreeRegressor()
        return dt
