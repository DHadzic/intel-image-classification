from abc import ABC, abstractmethod
from tensorflow import keras


class BaseModel(ABC):
    def __init__(self):
        self.input = None
        self.output = None
        self.define_layers()
        self.model = self.__define_model()

    @property
    @abstractmethod
    def epochs(self):
        pass

    @property
    @abstractmethod
    def validation_split(self):
        pass

    @abstractmethod
    def define_layers(self):
        pass

    def __define_model(self):
        model = keras.models.Model(inputs=self.input, outputs=self.output)

        return model

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
