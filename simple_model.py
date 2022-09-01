from datetime import date
from base_model import BaseModel
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten


class SimpleModel(BaseModel):
    def __init__(self):
        super().__init__()

    def _epochs(self):
        return 30

    def _validation_split(self):
        return 0.2

    def _define_layers(self):
        self.layers = [
            Conv2D(162, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPool2D(5, 5),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            Conv2D(104, kernel_size=(3, 3), activation='relu'),
            Conv2D(80, kernel_size=(3, 3), activation='relu'),
            Conv2D(56, kernel_size=(3, 3), activation='relu'),
            MaxPool2D(5, 5),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(92, activation='relu'),
            Dense(56, activation='relu'),
            Dropout(rate=0.5),
            Dense(6, activation='softmax')
        ]

    def _define_model(self):
        model = keras.models.Sequential(self.layers)

        return model

    def train_model(self, x_train, y_train):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print('Model overview:')
        print(self.model)

        self.model.fit(x_train, y_train, epochs=self._epochs(), validation_split=self._validation_split())

    def evaluate_on_data(self, x_test, y_test):
        _, accuracy = self.model.evaluate(x_test, y_test)

        print('Model accuracy on test > {} <'.format(accuracy))

    def save_model(self):
        date_str = date.today().strftime("%d-%m-%Y")
        model_file = 'models/cnn' + date_str
        self.model.save(model_file)
