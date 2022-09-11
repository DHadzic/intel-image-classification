from datetime import date
from base_model import BaseModel
from keras.layers import Conv2D, MaxPool2D, Dense, ZeroPadding2D, Dropout, Flatten, Input
from keras.models import load_model


class SimpleModel(BaseModel):
    def __init__(self):
        self.layers = None
        self.model_path1 = './models/simple-14k-71p'
        self.model_path2 = './models/simple-18k-70p'
        self.model_path3 = './models/simple-21k-70p'
        super().__init__()

    def epochs(self):
        return 24

    def validation_split(self):
        return 0.2

    def define_layers(self):
        self.input = Input((150, 150, 3))
        x = ZeroPadding2D((3, 3))(self.input)

        initial_filter = 128

        x = Conv2D(initial_filter, kernel_size=(7, 7), activation='relu', input_shape=(150, 150, 3))(x)
        x = Conv2D(initial_filter, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPool2D(5, 5)(x)
        x = Conv2D(initial_filter, kernel_size=(3, 3), activation='relu')(x)
        x = Conv2D(initial_filter * 2, kernel_size=(3, 3), activation='relu')(x)
        x = Conv2D(initial_filter * 4, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPool2D(5, 5)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(rate=0.5)(x)

        self.output = Dense(6, activation='softmax')(x)

    def train_model(self, x_train, y_train):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print('Model overview [Simple]:')
        print(self.model)

        self.model.fit(x_train, y_train, epochs=self.epochs(), validation_split=self.validation_split())

    def evaluate_on_data(self, x_test, y_test):
        _, accuracy = self.model.evaluate(x_test, y_test)

        print('Model accuracy on test > {} <'.format(accuracy))

    def save_model(self):
        date_str = date.today().strftime("%d-%m-%Y")
        model_file = 'models/cnn' + date_str
        self.model.save(model_file)

    def load_model(self, index=0):
        path = self.model_path1

        if index == 1:
            path = self.model_path2
        elif index == 2:
            path = self.model_path3

        self.model = load_model(path)

    def summary(self):
        self.model.summary()
