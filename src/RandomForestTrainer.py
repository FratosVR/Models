from ydf import RandomForestLearner
from ydf.Task import CLASSIFICATION


class RandomForestTrainer:

    def __init__(self):
        self.model = RandomForestLearner(label='pose', task=CLASSIFICATION)

    def train(self, X, y):
        X = X + y
        self.model.train(X)
