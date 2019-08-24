import pandas as pd
import numpy, matplotlib
from sklearn.linear_model import SGDRegressor, SGDClassifier
from scipy.stats import shapiro


class TestClass:
    def __init__(self, name, age, weight, rank=0, position="quarterback"):
        self.name = name
        self.age = age
        self.weight = weight
        self.rank = rank
        self.position = position

    def op(self):
        aw = self.age * self.weight
        return aw

    def op1(self, number1):
        return number1

    self.att1 = "attribute 1"


class Test2:
    """Example Doc STring"""


def function(weight, rank):
    wr = weight * rank
    return wr
