import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from tflearn.optimizers import Adam, Momentum, RMSProp


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=2.)

# img_aug.add_random_90degrees_rotation (rotations=[0, 1, 2, 3])
# img_aug.add_random_blur(sigma_max=5.0)

def get_network_architecture(image_width, image_height, number_of_classes, learning_rate):

    number_of_channels = 3

    network = input_data(
        shape=[None, image_width, image_height, number_of_channels],
        data_preprocessing=img_prep,
        data_augmentation=img_aug
    )

    print('\nLayers shape:')
    network = conv_2d(network, 32, (2, 2), 2, padding='same', activation='relu', name="Conv2D_1")
    print('  {}: {}'.format('Conv2D_1.............', network.shape))
    network = max_pool_2d(network, (2, 2), strides=2, padding='same', name="MaxPool2D_1")
    print('  {}: {}'.format('MaxPool2D_1..........', network.shape))

    network = conv_2d(network, 64, (2, 2), 2, padding='same', activation='relu', name="Conv2D_2")
    print('  {}: {}'.format('Conv2D_2.............', network.shape))
    network = max_pool_2d(network, (2, 2), strides=2, padding='same', name="MaxPool2D_2")
    print('  {}: {}'.format('MaxPool2D_2..........', network.shape))

    network = conv_2d(network, 128, (2, 2), 2, padding='same', activation='relu', name="Conv2D_3") # regularizer='L2'
    print('  {}: {}'.format('Conv2D_3.............', network.shape))
    network = max_pool_2d(network, (2, 2), strides=2, padding='same', name="MaxPool2D_3")
    print('  {}: {}'.format('MaxPool2D_3..........', network.shape))

    network = fully_connected(network, 256, activation='relu', name="FullyConnected_1")
    print('  {}: {}'.format('FullyConnected_1.....', network.shape))

    network = dropout(network, 0.5, name="Dropout")
    print('  {}: {}'.format('Dropout..............', network.shape))

    network = fully_connected(network, number_of_classes, activation='softmax', name="FullyConnected_Final")
    print('  {}: {}\n'.format('FullyConnected_Final.', network.shape))

    accuracy = Accuracy(name='Accuracy')

    # rmsprop = RMSProp(learning_rate=learning_rate, decay=0.9, momentum=0.4, epsilon=1e-10, name="RMSProp")
    adam = Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, name="Adam")

    network = regression(
        network,
        optimizer=adam,
        loss='softmax_categorical_crossentropy',
        metric=accuracy,
        learning_rate=learning_rate,
        name='Regression'
    )

    return network


if __name__ == '__main__':
    pass
