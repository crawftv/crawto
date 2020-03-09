#!/usr/bin/env python3
import functools


class MetaModel(object):
    def __init__(self, problem):
        self.problem = problem
        self.models = []

    def add_model_to_meta_model(self, model):
        self.models.append(model)

    def model(self, model):
        self.add_model_to_meta_model(model)

class Model(object):
    def __init__(self, model, problem):
        self.problem = problem
        self.model = model

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


