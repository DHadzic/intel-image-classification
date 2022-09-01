from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        self._define_layers()
        self.model = self._define_model()

    @property
    @abstractmethod
    def _epochs(self):
        pass

    @property
    @abstractmethod
    def _validation_split(self):
        pass

    @abstractmethod
    def _define_layers(self):
        pass

    @abstractmethod
    def _define_model(self):
        pass

    @abstractmethod
    def train_model(self, x_train, y_train):
        pass

    @abstractmethod
    def evaluate_on_data(self, x_test, y_test):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self, index=0):
        pass

    @abstractmethod
    def summary(self):
        pass
