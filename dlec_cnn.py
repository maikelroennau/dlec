from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.cross_validation import train_test_split
from glob import glob
import os

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy


def load_images(dataset_path, classes, number_of_images, number_of_channels):
    x = np.zeros((number_of_images, image_height, image_width, number_of_channels), dtype='float64')
    y = np.zeros(number_of_images)
    count = 0
    fails = 0

    print('\nLoading dataset')

    if os.path.isfile('x.npy') and os.path.isfile('y.npy'):
        print('Loading from backup')
        x = np.load('x.npy')
        y = np.load('y.npy')

        print('Dataset loaded from backup')
    else:
        for dataset_class in classes:
            print('Loading {} images'.format(dataset_class))

            images_path = os.path.join(dataset_path, dataset_class)
            for image in os.listdir(images_path):
                img = io.imread(os.path.join(images_path, image))

                # Reshape image if necessary
                reshaped_image = imresize(img, (image_height, image_width, number_of_channels))

                x[count] = np.array(reshaped_image)  # if reshaped: np.array(reshaped_image)
                y[count] = classes.index(dataset_class)
                count += 1

        print('\nSuccessful loaded {} images'.format(len(y)))
        print('Number of fails: {}'.format(fails))
        print('Saving arrays to disk')

        np.save('x', x)
        np.save('y', y)

    return x, y

def get_number_of_images(dataset_path):
    number_files = 0

    for image_class in os.listdir(dataset_path):
        number_files += len(os.listdir(os.path.join(dataset_path, image_class)))

    return number_files

def get_network_architecture(image_width, image_height, number_of_channels, img_prep, img_aug):
    network = input_data(
        shape=[None, image_height, image_width, number_of_channels],
        data_preprocessing=img_prep,
        data_augmentation=img_prep
    )

    conv_1 = conv_2d(network, 10, 11, 2, padding='same', activation='relu')
    
    max_pool_1 = mas_pool_1 = max_pool_2d(conv_1, 2, strides=2)

    fully_connected_1 = fully_connected(mas_pool_1, 500, activation='relu')
    
    fully_connected_2 = fully_connected(fully_connected_1, 250, activation='relu')

    dropout_1 = dropout(fully_connected_2, 0.4)

    fully_connected_3 = fully_connected(dropout_1, number_of_classes, activation='softmax')

    accuracy = Accuracy(name='Accuracy')

    fully_connected_3 = regression(
        fully_connected_3,
        optimizer='adam',
        loss='categorical_crossentropy',
        learning_rate=0.001,
        metric=accuracy
    )

    network = fully_connected_3

    return network


if __name__ == '__main__':
    dataset_path = 'dogs_vs_cats/train'
    train_logs_dir = 'training_logs/'

    model_checkpoint = 'model/'
    model_name = 'ck_model.tflearn'
    id = 'model_ck'

    image_width = 64
    image_height = 64
    number_of_channels = 3

    classes = os.listdir(dataset_path)
    number_of_classes = len(classes)
    number_of_images = get_number_of_images(dataset_path)

    print('\nClasses: ' + ', '.join(classes))
    print('Number of classes: {}'.format(number_of_classes))

    images, labels = load_images(dataset_path, classes, number_of_images, number_of_channels)

    X, X_test, Y, Y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

    Y = to_categorical(Y, number_of_classes)
    Y_test = to_categorical(Y_test, number_of_classes)

    img_prep = ImagePreprocessing()
    # img_prep.add_featurewise_zero_center()
    # img_prep.add_featurewise_stdnorm()

    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=2.)

    cnn = get_network_architecture(image_width, image_height, number_of_channels, img_prep, img_aug)

    model = tflearn.DNN(
        cnn,
        checkpoint_path=model_checkpoint,
        max_checkpoints=3,
        tensorboard_verbose=0,
        tensorboard_dir=train_logs_dir
    )

    model.fit(
        X, Y, 
        validation_set=(X_test, Y_test), 
        batch_size=200,
        n_epoch=1,
        run_id=id,
        show_metric=True
    )

    model.save(os.path.join(model_checkpoint, model_name))

    # Test

    img = io.imread('dogs_vs_cats/test/2.jpg')
    reshaped_image = imresize(img, (image_height, image_width, number_of_channels))
    to_predict = np.zeros((number_of_images, image_height, image_width, number_of_channels), dtype='float64')
    to_predict[0] = np.array(reshaped_image)  # if reshaped: np.array(reshaped_image)

    print(model.predict(to_predict))
    print(model.predict_label(to_predict))