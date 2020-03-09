#!/usr/bin/env python3
<<<<<<< HEAD
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


=======

class MetaModel(object):

    def __init__(
            problem
    ):
        self.problem = problem
>>>>>>> deb49cd8da39f2a4768f2cd229f4bb0e6befabe0
