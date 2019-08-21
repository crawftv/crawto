import pandas as pd
import numpy, matplotlib
from sklearn.linear_model import SGDRegressor,SGDClassifier
from scipy.stats import shapiro


class TestClass:
    def __init__(self, name, age, weight, rank = 0, position="quarterback"):
        self.name = name
        self.age = age
        self.weight = weight
        self.rank = rank
        self.position = position

    def op(self):
        aw = self.age * self.weight
        return aw

class Test2:
    """Example Doc STring"""


def function(weight, rank):
    wr = weight * rank
    return wr
