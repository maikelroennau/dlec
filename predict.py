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


def predict(model, images):
    cat = 0
    dog = 0

    i = 0
    for image in images:
        prediction = model.predict([image])[0]

        if prediction[0] > prediction[1]:
            imsave('predictions/cat/{}.jpg'.format(i), image)
        else:
            imsave('predictions/dog/{}.jpg'.format(i), image)

        print('cat: {}  dog: {}'.format(prediction[0], prediction[1]))
        i += 1


def save_image(image, name):
    imsave('predictions/{}.jpg'.format(name), image)


if __name__ == '__main__':
    model_path = 'final_model/final_model.tflearn'
    images_path = 'datasets/dogs_vs_cats/train/dog'

    image_height = 64
    image_width = 64
    learning_rate = 0.005
    number_of_classes = 2

    model = get_model(model_path, number_of_classes)
    model.load(model_path)

    images = load_images(images_path, image_height, image_width)
    
    predict(model, images)
    