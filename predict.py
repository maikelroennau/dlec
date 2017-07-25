import os
import numpy as np

import cnn
import dataset_loader

import tflearn
from tflearn.data_utils import to_categorical

from skimage import color, io
from scipy.misc import imresize, imsave


def get_model(model_path, number_of_classes):
    network = cnn.get_network_architecture(image_width, image_height, number_of_classes, learning_rate)
    model = tflearn.DNN(network)
    return model


def load_images(images_path, image_height, image_width):
    images = dataset_loader.load_images(images_path, image_height, image_width)
    return images


def get_list(array):
    new_list = []
    for item in array:
        new_list.append(item)

    return new_list


def load_classes(classes_path):
    return np.load(classes_path).tolist()


def show_predictions(model, images, classes):
    predictions = model.predict(images)

    for prediction_set in predictions:
        print('-' * 42)
        for j, prediction in enumerate(prediction_set):
            print('{:>9}:  {} \t{:>10}'.format(classes[j], prediction, '|'))
    print('-' * 42)


def separate(model, images):
    predictions = model.predict(images)

    print('Separating')
    for i, prediction_set in enumerate(predictions):
        if prediction_set[0] > prediction_set[1]:
            save_image('cat/{}'.format(i), images[i])
        else:
            save_image('dog/{}'.format(i), images[i])

def save_image(name, image):
    imsave('predictions/{}.jpg'.format(name), image)


if __name__ == '__main__':
    model_path = 'final_model/final_model.tflearn'
    images_path = 'datasets/dogs_vs_cats/validation'
    # images_path = 'datasets/jaffe_dataset/angriness'
    classes_path = 'data/dogs_vs_cats_classes.npy'

    image_height = 64
    image_width = 64
    learning_rate = 0.005

    classes = load_classes(classes_path)
    number_of_classes = len(classes)

    print('\nGetting model architecture\n')
    model = get_model(model_path, number_of_classes)
    
    print('\nLoading model')
    model.load(model_path)

    print('Loading imges')
    images = load_images(images_path, image_height, image_width)
    
    print('Predictions')
    # show_predictions(model, images, classes)
    separate(model, images)
