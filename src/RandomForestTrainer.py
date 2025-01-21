from sklearn.ensemble import RandomForestClassifier


class RandomForestTrainer:

    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)
