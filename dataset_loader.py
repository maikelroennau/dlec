import os
import numpy as np

import cv2

from glob import glob

from skimage import color, io
from scipy.misc import imresize, imsave

from tflearn.data_utils import image_preloader


def load_dataset_images(dataset_path, image_height, image_width, dataset_name='unnamed', colored=True, load_backup=False, export_dataset=False):

    number_of_channels = 3

    classes = get_classes(dataset_path)
    print('\nDataset name: {}'.format(dataset_name))
    print('\nClasses: {}'.format(', '.join(classes)))
    print('Number of classses: {}'.format(len(classes)))

    number_of_images, images_per_class = get_total_number_of_images(dataset_path)

    print('\nNumber of images per class:')
    for key, value in images_per_class.iteritems():
        print('{}: {}'.format(key, images_per_class.get(key)))

    print('\nTotal images: {}'.format(number_of_images))

    # if colored:
    x = np.zeros((number_of_images, image_height, image_width, number_of_channels), dtype='float64')
    # else:
        # x = np.zeros((number_of_images, image_height, image_width), dtype='float64')

    y = np.zeros(number_of_images)
    count = 0
    fails = 0

    print('\nLoading dataset')

    data_folder = 'data/'

    if load_backup and (os.path.isfile('{}{}_x.npy'.format(data_folder, dataset_name)) and os.path.isfile('{}{}_y.npy'.format(data_folder, dataset_name))):
        print('Loading from backup')
        x = np.load('{}{}_x.npy'.format(data_folder, dataset_name))
        y = np.load('{}{}_y.npy'.format(data_folder, dataset_name))

        print('Dataset loaded from backup')
    else:
        for dataset_class in classes:
            print('Loading {} class'.format(dataset_class))

            images_path = os.path.join(dataset_path, dataset_class)
            for image in os.listdir(images_path):
                img = cv2.imread(os.path.join(images_path, image))

                # if colored:
                reshaped_image = imresize(img, (image_height, image_width, number_of_channels))
                # else:
                    # reshaped_image = imresize(img, (image_height, image_width))

                x[count] = np.array(reshaped_image)
                y[count] = classes.index(dataset_class)

                count += 1

        print('\nSuccessful loaded {} images'.format(len(y)))
        print('Number of fails: {}'.format(fails))
        print('Saving arrays to disk')

        if export_dataset:
            if not os.path.exists('data'):
                os.makedirs('data')
            np.save('{}{}_x'.format(data_folder, dataset_name), x)
            np.save('{}{}_y'.format(data_folder, dataset_name), y)
            np.save('{}{}_classes'.format(data_folder, dataset_name), get_classes_dictionary(classes))

    return x, y


def load_images(images_path, image_height, image_width, colored=True):

    number_of_images = len(os.listdir(images_path))

    if colored:
        number_of_channels = 3
    else:
        number_of_channels = 1

    print('\nLoading images')

    x = np.zeros((number_of_images, image_height, image_width, number_of_channels), dtype='float64')
    count = 0
    fails = 0

    for image in os.listdir(images_path):
        img = cv2.imread(os.path.join(images_path, image))

        reshaped_image = imresize(img, (image_height, image_width, number_of_channels))

        x[count] = np.array(reshaped_image)
        count += 1

    print('\nSuccessful loaded {} images'.format(len(x)))
    print('Number of fails: {}\n'.format(fails))

    return x


def get_classes(dataset_path):
    return os.listdir(dataset_path)


def get_classes_dictionary(classes):
    ditcionary = {}
    
    for i, class_name in enumerate(classes):
        ditcionary[i] = class_name

    return ditcionary


def get_total_number_of_images(dataset_path):
    total_images = 0
    images_per_class = {}

    i = 0
    for dataset_class in os.listdir(dataset_path):
        class_images = len(os.listdir(os.path.join(dataset_path, dataset_class)))

        total_images += class_images
        images_per_class[dataset_class] = class_images

    return total_images, images_per_class


if __name__ == '__main__':
    pass
