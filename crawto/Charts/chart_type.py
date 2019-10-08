from typing import List


class DataPoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class DataSet:
    def __init__(self, label: str, data: List[DataPoint], backgroundColor: str):
        self.label = label
        self.data = data
        self.backgroundColor = backgroundColor


class Data:
    def __init__(self, datasets: List[DataSet] = []):
        self.datasets = datasets
