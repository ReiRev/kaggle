from abc import abstractmethod
# from ..evaluation import cross_validation_score
from sklearn.metrics import accuracy_score


class Model:
    def __init__(self, params: dict):
        self.params = params
        self.model = None
        self.scaler = None

    @abstractmethod
    def fit(self, X_train, y_train) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_pred):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__
