import os
import numpy as np

import cnn
import dataset_loader

import tflearn
from tflearn.data_utils import to_categorical

from skimage import color, io
from scipy.misc import imread, imresize, imsave


def get_model(model_path, number_of_classes):
    network = cnn.get_network_architecture(image_width, image_height, number_of_classes, learning_rate)
    model = tflearn.DNN(network)
    return model

def load_images(images_path, image_width, image_height):
    images = dataset_loader.load_images(images_path, image_width, image_height)
    return images

def load_image(images_path, image_width, image_height):
    image = dataset_loader.load_image(images_path, image_width, image_height)
    return image

def load_classes(classes_path):
    return np.load(classes_path).tolist()

def get_list(array):
    new_list = []
    for item in array:
        new_list.append(item)

    return new_list

def save_image(name, image):
    imsave('predictions/{}.jpg'.format(name), image)

def predict(model, images, classes):
    predictions = model.predict(images)

    for prediction_set in predictions:
        print('-' * 42)
        for j, prediction in enumerate(prediction_set):
            print('{:>9}:  {} \t{:>10}'.format(classes[j], prediction, '|'))
    print('-' * 42)

def predict_image(model, image, classes):
    predictions = model.predict(image)

    print('-' * 42)
    for j, prediction in enumerate(predictions):
        print('{:>9}:  {} \t{:>10}'.format(classes[j], prediction, '|'))
    print('-' * 42)


if __name__ == '__main__':
    model_path = 'final_model/final_model.tflearn'
    images_path = 'datasets/dogs_vs_cats/validation'
    classes_path = 'data/dogs_vs_cats_classes.npy'

    image_width = 64
    image_height = 64
    learning_rate = 0.005

    classes = load_classes(classes_path)
    number_of_classes = len(classes)

    print('\nGetting model architecture')
    model = get_model(model_path, number_of_classes)
    
    print('\nLoading model')
    model.load(model_path)

    images = load_images(images_path, image_width, image_height)
    # image = load_image('/home/maikel/dlec/datasets/dogs_vs_cats/cars/car_0.jpg', image_width, image_height)
    
    print('Predictions')
    # predict_image(model, image, classes)
    predict(model, images, classes)
