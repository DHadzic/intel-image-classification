from data_helper import DataHelper
from simple_model import SimpleModel
from resnet_model import ResNetModel
from inception_model import InceptionModel

options = {
    # If False, will do load data instead
    'execute_train': True,
    # Only works if execute_train is False
    'model_index': 0,
    'remove_noise': False,
    'create_noise': False,
    'show_summary': False,
}

if __name__ == '__main__':
    data_helper = DataHelper()

    if options['remove_noise']:
        # Only call when required
        data_helper.remove_noise_set()

    model = SimpleModel()
    # model = ResNetModel()
    # model = InceptionModel()

    if options['execute_train']:
        train_path = './data/seg_train/'
        train_pics = data_helper.load_picture_paths(train_path)

        data_helper.remove_different(train_pics)
        if options['create_noise']:
            # Only call when required
            data_helper.create_noise_set(train_pics)

        data_helper.analyze_pictures('train', train_pics)
        train_pics = data_helper.load_and_resize(train_pics)

        x_train, y_train = data_helper.create_xy(train_pics)
        model.train_model(x_train, y_train)
        model.save_model()
    else:
        model.load_model(options['model_index'])

    test_path = './data/seg_test/'
    test_pics = data_helper.load_picture_paths(test_path)
    data_helper.analyze_pictures('test', test_pics)
    data_helper.remove_different(test_pics)
    test_pics = data_helper.load_and_resize(test_pics)
    x_test, y_test = data_helper.create_xy(test_pics)

    if options['show_summary']:
        model.summary()

    model.evaluate_on_data(x_test, y_test)
