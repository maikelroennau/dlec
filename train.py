import os

import cnn
import dataset_loader

import tflearn
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import shuffle, to_categorical


def train():
    train_dataset = 'datasets/dogs_vs_cats/train'
    validation_dataset = 'dataset/validation'

    number_of_classes = len(dataset_loader.get_classes(train_dataset))
    learning_rate = 0.001
    
    image_width = 68
    image_height = 68
    number_of_channels = 3

    images, labels = dataset_loader.load_images(train_dataset, image_height, image_width, colored=True)

    X, X_test, Y, Y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

    Y = to_categorical(Y, number_of_classes)
    Y_test = to_categorical(Y_test, number_of_classes)

    model = cnn.get_network_architecture(image_width, image_height, number_of_channels, number_of_classes, learning_rate)

    # if not os.path.isdir('checkpoints'):
        # os.mkdir('checkpoints')
        # os.mkdir('checkpoints/best')

    model = tflearn.DNN(
        model,
        tensorboard_verbose=0,
        tensorboard_dir='train_logs/',
        # checkpoint_path='checkpoints/',
        # best_checkpoint_path='checkpoints/best/',
        max_checkpoints=2
    )

    model.fit(
        X, Y,
        validation_set=(X_test, Y_test),
        batch_size=200,
        n_epoch=3,
        run_id='001',
        show_metric=True
    )

    if not os.path.isdir('final_model/'):
        os.mkdir('final_model')
    
    print('Saving trained model to {}'.format('model'))
    model.save('final_model/final_model.tflearn')

    del images, labels, X, X_test, Y, Y_test

if __name__ == '__main__':
    train()