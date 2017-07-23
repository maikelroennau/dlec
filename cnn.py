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


def get_network_architecture(image_width, image_height, number_of_channels, number_of_classes, learning_rate):

    network = input_data(
        shape=[None, image_height, image_width, number_of_channels],
        data_preprocessing=img_prep,
        data_augmentation=img_aug
    )

    # network = conv_2d(network, 32, 3, activation='relu', name="Conv2D_1")
    # network = max_pool_2d(network, 2)
    # network = conv_2d(network, 64, 3, activation='relu', name="Conv2D_2")
    # network = max_pool_2d(network, 2, strides=2)
    # network = fully_connected(network, 512, activation='relu')
    # network = fully_connected(network, 250, activation='relu')

    network = conv_2d(network, 32, (5, 5), 2, padding='same', activation='relu', name="Conv2D_1")
    network = max_pool_2d(network, (2, 2), strides=2, padding='same', name="MaxPool2D_1")

    network = conv_2d(network, 62, (5, 5), 2, padding='same', activation='relu', name="Conv2D_2")
    network = max_pool_2d(network, (2, 2), strides=2, padding='same', name="MaxPool2D_2")

    network = fully_connected(network, 512, activation='relu', name="FullyConnected__1")
    network = fully_connected(network, 250, activation='relu', name="FullyConnected_2")
    
    network = dropout(network, 0.4, name="Dropout")
    network = fully_connected(network, number_of_classes, activation='softmax', name="FullyConnected_Final")

    network = regression(
        network,
        optimizer='adam',
        loss='categorical_crossentropy',
        metric='default',
        learning_rate=learning_rate,
        name='Regression'
    )

    # accuracy = Accuracy(name='Accuracy')
    # adam = tflearn.optimizers.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name="Adam")

    return network


if __name__ == '__main__':
    pass
