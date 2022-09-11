from datetime import date
from base_model import BaseModel
from keras.layers import Dropout, Dense, AveragePooling2D, Flatten, Input
from keras.models import load_model
from keras.applications.resnet import ResNet50


# noinspection PyMethodMayBeStatic
class ResNetModel(BaseModel):
    def __init__(self):
        # self.resnet_model = ResNet152(weights="imagenet", include_top=False, input_tensor=Input(shape=(150, 150, 3)))
        self.resnet_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(150, 150, 3)))
        self.model_path1 = './models/resnet-fine50-14k-79p'
        self.model_path2 = './models/resnet-fine50-21k-78p'
        self.model_path3 = './models/resnet-fine152-21k-80p'
        super().__init__()

    def epochs(self):
        return 10

    def validation_split(self):
        return 0.2

    def define_layers(self):
        train_layers_no = 35
        for i, layer in enumerate(self.resnet_model.layers):
            if i > len(self.resnet_model.layers) - train_layers_no:
                break
            layer.trainable = False

        self.input = self.resnet_model.input
        x = AveragePooling2D(pool_size=(7, 7), padding="same")(self.resnet_model.output)
        x = Flatten(name="flatten")(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        self.output = Dense(6, activation="softmax")(x)

    def train_model(self, x_train, y_train):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print('Model overview [ResNet]:')
        print(self.model)

        self.model.fit(x_train, y_train, epochs=self.epochs(), validation_split=self.validation_split())

    def evaluate_on_data(self, x_test, y_test):
        _, accuracy = self.model.evaluate(x_test, y_test)

        print('Model accuracy on test > {} <'.format(accuracy))

    def save_model(self):
        date_str = date.today().strftime("%d-%m-%Y")
        model_file = 'models/resnet' + date_str
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

    # Method used for building of Residual blocks
    """
    self.input = Input((150, 150, 3))
    x = ZeroPadding2D((3, 3))(self.input)
    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    block_layers = [3, 4, 6, 3]
    for _ in range(block_layers[0]):
        x = self.res_block(x, 64)
    x = self.res_block(x, 128, True)
    for _ in range(block_layers[1] - 1):
        x = self.res_block(x, 128)
    x = self.res_block(x, 256, True)
    for _ in range(block_layers[2] - 1):
        x = self.res_block(x, 256)
    x = self.res_block(x, 512, True)
    for _ in range(block_layers[3] - 1):
        x = self.res_block(x, 512)
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    self.output = Dense(6, activation='softmax')(x)

    def res_block(self, x, filter_size, is_conv=False):
        x_skip = x

        strides = (2, 2) if is_conv else (1, 1)

        x = Conv2D(filter_size, (3, 3), padding='same', strides=strides)(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filter_size, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        if is_conv:
            x_skip = Conv2D(filter_size, (1, 1), strides=strides)(x_skip)

        x = Add()([x, x_skip])
        x = Activation('relu')(x)
        return x
    """