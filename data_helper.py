import glob
import numpy as np
import matplotlib.pyplot as plot
import cv2
import os
import random
from sklearn.metrics import confusion_matrix


# noinspection PyMethodMayBeStatic
class DataHelper:
    def __init__(self):
        self.code = {'mountain': 0, 'street': 1, 'glacier': 2, 'buildings': 3, 'sea': 4, 'forest': 5}

    def display_samples(self, pictures, size=150):
        images = []
        labels = []
        for key in pictures.keys():
            for picture in pictures[key]:
                image = cv2.imread(picture)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (size, size))
                images.append(image)
                labels.append(key)

        image = np.array(image)
        labels = np.array(labels)

        f, ax = plot.subplots(5, 5)
        f.subplots_adjust(left=0, bottom=0, right=0.8, top=0.8)
        for i in range(0, 5, 1):
            for j in range(0, 5, 1):
                rnd = random.randint(0, len(images))
                ax[i, j].imshow(images[rnd])
                ax[i, j].set_title(labels[rnd])
                ax[i, j].axis('off')

        plot.show()

    def display_confiusion_matrix(self, model_obj, x_data, y_data):
        predictions = model_obj.model.predict(x_data)
        pred_data = np.argmax(predictions, axis=1)

        conf_matrix = confusion_matrix(y_data, pred_data)

        fig, ax = plot.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plot.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plot.xticks(np.arange(6), self.code.keys())
        plot.yticks(np.arange(6), self.code.keys())
        plot.xlabel('Predictions', fontsize=18)
        plot.ylabel('Actuals', fontsize=18)
        plot.title('Confusion Matrix', fontsize=18)
        plot.show()

    def create_noise_image(self, path, image_class, index, image):
        if 'seg_train' not in path:
            return
        file = path.split('/').pop()
        file = file.split('.')[0] + '_noise.' + file.split('.')[1]
        save_path = './data/seg_train/' + image_class + '/' + file

        # Do it only for 1/2 of the set
        if index % 8 == 0:
            modified_image = self.noisy('zoom', image)
        elif index % 8 == 1:
            modified_image = self.noisy('flip', image)
        elif index % 8 == 2:
            modified_image = self.noisy('brightness', image)
        elif index % 8 == 3:
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
        if noise_typ == "zoom":
            zoom_index = 0.8
            h, w = image.shape[:2]
            h_taken = int(zoom_index * h)
            w_taken = int(zoom_index * w)
            h_start = random.randint(0, h - h_taken)
            w_start = random.randint(0, w - w_taken)
            image = image[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
            noisy_img = cv2.resize(image, (h, w), cv2.INTER_CUBIC)
            return noisy_img
        elif noise_typ == "flip":
            noisy_img = cv2.flip(image, 1)
            return noisy_img
        elif noise_typ == "brightness":
            brightness_index = 0.75
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 1] = hsv[:, :, 1] * brightness_index
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            hsv[:, :, 2] = hsv[:, :, 2] * brightness_index
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            hsv = np.array(hsv, dtype=np.uint8)
            noisy_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return noisy_img
        elif noise_typ == "speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy_img = image + image * (gauss * 0.1)
            return noisy_img

    def create_noise_set(self, pictures):
        for key in pictures.keys():
            # There are already noise images created
            if len(pictures[key]) > 2500:
                continue
            for index, picture in enumerate(pictures[key]):
                image = cv2.imread(picture)
                self.create_noise_image(picture, key, index, image)

    def remove_noise_set(self):
        for class_label in self.code.keys():
            dir_name = './data/seg_train/' + class_label + '/'
            dir_items = os.listdir(dir_name)
            for item in dir_items:
                if item.endswith("noise.jpg"):
                    os.remove(os.path.join(dir_name, item))

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

        print(y_data.count(0))
        print(y_data.count(1))
        print(y_data.count(2))
        print(y_data.count(3))
        print(y_data.count(4))
        print(y_data.count(5))
        return np.array(x_data), np.array(y_data)
