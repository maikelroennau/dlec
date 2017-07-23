import os
import numpy as np
from glob import glob
from skimage import color, io
from scipy.misc import imresize, imsave


def load_images(dataset_path, image_height, image_width, colored=True):
    
    classes = get_classes(dataset_path)
    print('\nClasses: {}'.format(', '.join(classes)))
    print('Number of classses: {}'.format(len(classes)))

    number_of_images, images_per_class = get_total_number_of_images(dataset_path)

    print('\nNumber of images per class:')
    for key, value in images_per_class.iteritems():
        print('{}: {}'.format(key, images_per_class.get(key)))

    print('\nTotal images: {}'.format(number_of_images))
    
    if colored:
        number_of_channels = 3
    else:
        number_of_channels = 1

    x = np.zeros((number_of_images, image_height, image_width, number_of_channels), dtype='float64')
    y = np.zeros(number_of_images)
    count = 0
    fails = 0

    print('\nLoading dataset')

    if os.path.isfile('arrays/x.npy') and os.path.isfile('arrays/y.npy'):
        print('Loading from backup')
        x = np.load('arrays/x.npy')
        y = np.load('arrays/y.npy')

        print('Dataset loaded from backup')
    else:
        for dataset_class in classes:
            print('Loading {} class'.format(dataset_class))

            images_path = os.path.join(dataset_path, dataset_class)
            for image in os.listdir(images_path):
                img = io.imread(os.path.join(images_path, image))

                reshaped_image = imresize(img, (image_height, image_width, number_of_channels))

                x[count] = np.array(reshaped_image)
                y[count] = classes.index(dataset_class)

                count += 1

        print('\nSuccessful loaded {} images'.format(len(y)))
        print('Number of fails: {}'.format(fails))
        print('Saving arrays to disk')

        if not os.path.exists('arrays'):
            os.makedirs('arrays')
        np.save('arrays/x', x)
        np.save('arrays/y', y)

    return x, y


def get_classes(dataset_path):
    return os.listdir(dataset_path)

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
