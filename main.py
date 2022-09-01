from datetime import date
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from data_helper import DataHelper
from simple_model import SimpleModel

if __name__ == '__main__':
    data_helper = DataHelper()

    train_path = './data/seg_train/'
    test_path = './data/seg_test/'
    pred_path = './data/seg_pred/'

    train_pics = data_helper.load_picture_paths(train_path)
    test_pics = data_helper.load_picture_paths(test_path)

    data_helper.analyze_pictures('train', train_pics)
    data_helper.analyze_pictures('test', test_pics)

    data_helper.remove_different(train_pics)
    data_helper.remove_different(test_pics)

    # Only call when required
    # data_helper.create_noise_set(train_pics);

    train_pics = data_helper.load_and_resize(train_pics)
    test_pics = data_helper.load_and_resize(test_pics)

    x_train, y_train = data_helper.create_xy(train_pics)
    x_test, y_test = data_helper.create_xy(test_pics)

    model = SimpleModel()

    model.train_model(x_train, y_train)

    model.save_model()

    model.evaluate_on_data(x_test, y_test)

