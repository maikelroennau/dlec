import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization, local_response_normalization
from tflearn.metrics import Accuracy

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from tflearn.optimizers import Adam, Momentum, RMSProp


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)


def get_network_architecture(image_width, image_height, number_of_classes, learning_rate):

    number_of_channels = 3

    network = input_data(
        shape=[None, image_width, image_height, number_of_channels],
        data_preprocessing=img_prep,
        data_augmentation=img_aug
    )

    """
        def conv_2d(incoming, nb_filters, filter_size, strides=1, padding='same',
                    activation='linear', bias='True', weights_init='uniform_scaling',
                    bias_init='zeros', regularizer=None, weight_decai=0.001,
                    trainable=True, restore=True, reuse=False, scope=None,
                    name='Conv2D')

        def max_pool_2d(incoming, kernel_size, strides=None,
                        padding='same', name='MaxPool2D")
    """

    print('\nLayers shape:')
    network = conv_2d(network, 32, 6, strides=1, padding='same', activation='relu', regularizer='L2', name="Conv2D_1")
    print('  {}: {}'.format('Conv2D................', network.shape))
    network = max_pool_2d(network, 4, strides=2, padding='same', name="MaxPool2D_1")
    print('  {}: {}'.format('MaxPool2D.............', network.shape))

    network = dropout(network, 0.3, name="Dropout_1")
    print('  {}: {}'.format('Dropout...............', network.shape))
    network = batch_normalization(network, name='BatchNormalization_1')
    print('  {}: {}'.format('BatchNormalization....', network.shape))


    network = conv_2d(network, 64, 4, strides=1, padding='same', activation='relu', regularizer='L2', name="Conv2D_2")
    print('  {}: {}'.format('Conv2D................', network.shape))
    network = max_pool_2d(network, 4, strides=2, padding='same', name="MaxPool2D_2")
    print('  {}: {}'.format('MaxPool2D.............', network.shape))

    network = dropout(network, 0.3, name="Dropout_2")
    print('  {}: {}'.format('Dropout...............', network.shape))
    network = batch_normalization(network, name='BatchNormalization_2')
    print('  {}: {}'.format('BatchNormalization....', network.shape))


    network = conv_2d(network, 128, 2, strides=1, padding='same', activation='relu', regularizer='L2', name="Conv2D_3")
    print('  {}: {}'.format('Conv2D................', network.shape))
    network = max_pool_2d(network, 2, strides=2, padding='same', name="MaxPool2D_3")
    print('  {}: {}'.format('MaxPool2D.............', network.shape))
    
    network = dropout(network, 0.3, name="Dropout_3")
    print('  {}: {}'.format('Dropout...............', network.shape))
    network = batch_normalization(network, name='BatchNormalization_3')
    print('  {}: {}'.format('BatchNormalization....', network.shape))


    network = conv_2d(network, 256, 2, strides=2, padding='same', activation='relu', regularizer='L2', name="Conv2D_4")
    print('  {}: {}'.format('Conv2D_2..............', network.shape))
    network = max_pool_2d(network, 2, strides=2, padding='same', name="MaxPool2D_4")
    print('  {}: {}'.format('MaxPool2D.............', network.shape))

    network = batch_normalization(network, name='BatchNormalization_4')
    print('  {}: {}'.format('BatchNormalization....', network.shape))
    network = dropout(network, 0.3, name="Dropout_4")
    print('  {}: {}'.format('Dropout...............', network.shape))


    network = fully_connected(network, 4096, activation='relu', name="FullyConnected_1")
    print('  {}: {}'.format('FullyConnected........', network.shape))

    network = fully_connected(network, number_of_classes, activation='softmax', name="FullyConnected_Final")
    print('  {}: {}\n'.format('FullyConnected_Final.', network.shape))

    # accuracy = Accuracy(name='Accuracy')

    # rmsprop = RMSProp(learning_rate=learning_rate, decay=0.9, momentum=0.4, epsilon=1e-10, name="RMSProp")
    # adam = Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, name="Adam")

    network = regression(
        network,
        optimizer='momentum',
        loss='categorical_crossentropy',
        metric='accuracy',
        learning_rate=learning_rate,
        name='Regression'
    )

    return network


if __name__ == '__main__':
    pass
