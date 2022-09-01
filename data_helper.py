import glob
import numpy as np
import matplotlib.pyplot as plot
import cv2
import os


# noinspection PyMethodMayBeStatic
class DataHelper:
    def __init__(self):
        self.code = {'mountain': 0, 'street': 1, 'glacier': 2, 'buildings': 3, 'sea': 4, 'forest': 5}

    def create_noise_image(self, path, image_class, index, image):
        if 'seg_train' not in path:
            return
        file = path.split('/').pop()
        file = file.split('.')[0] + '_noise.' + file.split('.')[1]
        save_path = './data/seg_train/' + image_class + '/' + file
        # Do it only for 1/3 of the set
        if index % 12 == 0:
            modified_image = self.noisy('gauss', image)
        elif index % 12 == 1:
            modified_image = self.noisy('s&p', image)
        elif index % 12 == 2:
            modified_image = self.noisy('poisson', image)
        elif index % 12 == 3:
            modified_image = self.noisy('speckle', image)
        else:
            return
        cv2.imwrite(save_path, modified_image)

    def load_picture_paths(self, path):
        ret_value = {}
        for folder in os.listdir(path):
            ret_value[folder] = glob.glob(pathname=str(path + folder + '/*.jpg'))
        return ret_value

    def noisy(self, noise_typ, image):
        if noise_typ == "gauss":
            row, col, ch = image.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy_img = image + gauss
            return noisy_img
        elif noise_typ == "s&p":
            _, _, _ = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy_img = np.random.poisson(image * vals) / float(vals)
            return noisy_img
        elif noise_typ == "speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy_img = image + image * gauss
            return noisy_img

    def create_noise_set(self, pictures):
        for key in pictures.keys():
            # There are already noise images created
            if len(pictures[key]) > 2500:
                continue
            for index, picture in enumerate(pictures[key]):
                image = cv2.imread(picture)
                self.create_noise_image(picture, key, index, image)

    def load_and_resize(self, pictures, size=150):
        resized_pictures = {}
        for key in pictures.keys():
            current_array = []
            for index, picture in enumerate(pictures[key]):
                image = cv2.imread(picture)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (size, size))
                current_array.append(list(image))
            resized_pictures[key] = current_array
        return resized_pictures

    def analyze_pictures(self, label, pictures):
        print('~' * 20)
        print('Analyzing data from group > ' + label + ' <')
        for key in pictures.keys():
            print('Group > {:12s} < contains > {:7s} < images'.format(key, str(len(pictures[key]))))
        print('~' * 20)

    def remove_different(self, pictures):
        total_pictures = sum([len(pictures[key]) for key in pictures.keys()])
        print('~' * 20)
        print('Removing all non (150,150) images..')
        print('Number of images before removal: ' + str(total_pictures))
        for key in pictures.keys():
            to_remove = []
            for file in pictures[key]:
                image = plot.imread(file)
                if image.shape != (150, 150, 3):
                    to_remove.append(file)
            pictures[key] = [item for item in pictures[key] if item not in to_remove]
        total_pictures = sum([len(pictures[key]) for key in pictures.keys()])
        print('Number of images after removal: ' + str(total_pictures))
        print('~' * 20)

    def create_xy(self, pictures):
        x_data = []
        y_data = []
        for key in pictures.keys():
            for picture in pictures[key]:
                x_data.append(picture)
                y_data.append(self.code[key])
        return np.array(x_data), np.array(y_data)
