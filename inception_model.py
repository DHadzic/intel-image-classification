from datetime import date
from base_model import BaseModel
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3


# noinspection PyMethodMayBeStatic
class InceptionModel(BaseModel):
    def __init__(self):
        self.inception_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(150, 150, 3)))
        self.model_path1 = './models/inception-fine-21k-71p'
        super().__init__()

    def epochs(self):
        return 20

    def validation_split(self):
        return 0.25

    def define_layers(self):
        train_layers_no = 270
        for i, layer in enumerate(self.inception_model.layers):
            if i > len(self.inception_model.layers) - train_layers_no:
                break
            layer.trainable = False

        self.input = self.inception_model.input

        x = GlobalAveragePooling2D()(self.inception_model.output)
        x = Dense(512, activation='relu')(x)

        self.output = Dense(6, activation="softmax")(x)

    def train_model(self, x_train, y_train):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print('Model overview [Inception]:')
        print(self.model)

        self.model.fit(x_train, y_train, epochs=self.epochs(), validation_split=self.validation_split())

    def evaluate_on_data(self, x_test, y_test):
        _, accuracy = self.model.evaluate(x_test, y_test)

        print('Model accuracy on test > {} <'.format(accuracy))

    def save_model(self):
        date_str = date.today().strftime("%d-%m-%Y")
        model_file = 'models/inception' + date_str
        self.model.save(model_file)

    def load_model(self, index=0):
        path = self.model_path1

        self.model = load_model(path)

    def summary(self):
        self.model.summary()
