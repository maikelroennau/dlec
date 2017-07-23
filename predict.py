import numpy as np

import cnn
import dataset_loader

import tflearn

from skimage import color, io
from scipy.misc import imresize, imsave


train_dataset = 'datasets/dogs_vs_cats/train'
validation_dataset = 'datasets/dogs_vs_cats/small'

number_of_classes = len(dataset_loader.get_classes(train_dataset))
learning_rate = 0.001

image_width = 68
image_height = 68
number_of_channels = 3

model_path = 'final_model/final_model.tflearn'

network = cnn.get_network_architecture(image_width, image_height, number_of_channels, number_of_classes, learning_rate)

model = tflearn.DNN(network)
model.load(model_path)

image_path = '/home/maikel/Desktop/dlec/datasets/dogs_vs_cats/small/1.jpg'

image = io.imread(image_path)#.reshape((image_width, image_height, number_of_channels)).astype(np.float) / 255
image = imresize(image, (image_height, image_width, number_of_channels)).astype(np.float) / 255

print('\n{}'.format(dataset_loader.get_classes(train_dataset)))
print(model.predict_label([image]))


if __name__ == '__main__':
    pass
