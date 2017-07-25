import os

import cnn
import dataset_loader

import tflearn
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import shuffle, to_categorical


def train():
    train_dataset = 'datasets/dogs_vs_cats/train'
    dataset_name = train_dataset.split('/')[1]
    validation_dataset = None

    number_of_classes = len(dataset_loader.get_classes(train_dataset))
    learning_rate = 0.005

    image_width = 64
    image_height = 64

    images, labels = dataset_loader.load_dataset_images(train_dataset, image_height, image_width, dataset_name, colored=True, load_backup=True, export_dataset=True)
    X, X_test, Y, Y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

    Y = to_categorical(Y, number_of_classes)
    Y_test = to_categorical(Y_test, number_of_classes)

    model = cnn.get_network_architecture(image_height, image_width, number_of_classes, learning_rate, colored=True)

    model = tflearn.DNN(
        model,
        tensorboard_verbose=0,
        tensorboard_dir='train_logs/',
        max_checkpoints=2
    )

    model.fit(
        X, Y,
        validation_set=(X_test, Y_test),
        batch_size=int((len(X)*0.01)),
        n_epoch=10,
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
