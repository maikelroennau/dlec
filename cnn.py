import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=2.)

# img_aug.add_random_90degrees_rotation (rotations=[0, 1, 2, 3])
# img_aug.add_random_blur(sigma_max=5.0)


def get_network_architecture(image_width, image_height, number_of_classes, learning_rate, colored=True):

    number_of_channels = 3

    if colored:
        network = input_data(
            shape=[None, image_height, image_width, number_of_channels],
            data_preprocessing=img_prep,
            data_augmentation=img_aug
        )
    else:
        network = input_data(
            shape=[None, image_height, image_width],
            data_preprocessing=img_prep,
            data_augmentation=img_aug
        )

    network = conv_2d(network, 32, (6, 6), 2, padding='same', activation='relu', name="Conv2D_1")
    network = max_pool_2d(network, (2, 2), strides=2, padding='same', name="MaxPool2D_1")

    network = conv_2d(network, 64, (3, 3), 2, padding='same', activation='relu', name="Conv2D_2")
    network = max_pool_2d(network, (2, 2), strides=2, padding='same', name="MaxPool2D_2")

    network = conv_2d(network, 128, (2, 2), 2, padding='same', activation='relu', name="Conv2D_3", regularizer='L2')
    network = max_pool_2d(network, (2, 2), strides=2, padding='same', name="MaxPool2D_3")

    network = fully_connected(network, 256, activation='relu', name="FullyConnected__1")

    network = dropout(network, 0.5, name="Dropout")

    network = fully_connected(network, number_of_classes, activation='softmax', name="FullyConnected_Final")

    accuracy = Accuracy(name='Accuracy')

    network = regression(
        network,
        optimizer='adam',
        loss='softmax_categorical_crossentropy',
        metric=accuracy,
        learning_rate=learning_rate,
        name='Regression'
    )

    return network


if __name__ == '__main__':
    pass
