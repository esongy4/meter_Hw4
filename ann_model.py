from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from data_preprocessing import DataPreprocessing


class ANNModel(DataPreprocessing):

    def __init__(self, filepath):
        super().__init__(filepath)

    def train_model(self, X_train, y_train):
        self.model = MLPClassifier(
            hidden_layer_sizes=(20,10),
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def cross_validate(self, X, y):
        return cross_val_score(self.model, X, y, cv=5)