import os
import shutil

import tflearn
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import shuffle, to_categorical

import cnn
import dataset_loader


def create_directories():
    directories = ['best_checkpoint', 'train_logs', 'final_model']

    for directory in directories:
        try:
            shutil.rmtree(directory)
            os.makedirs(directory)
        except OSError:
            os.makedirs(directory)

def get_model(model_path, model):
    print('\nRestoring model')
    model.load(model_path)

    return model

def backup_network():
    shutil.copyfile('cnn.py', 'final_model/cnn.py')
    open('final_model/__init__.py', 'a')

def get_batch_size(number_of_images, percentage):
    if (number_of_images * percentage) > 128:
        return 128
    else:
        return int(number_of_images * percentage)

def train(resume_training=False):
    train_dataset = 'Datasets/CK'
    train_dataset_name = train_dataset.split('/')[1]
    validation_dataset = 'Datasets/Custom'

    if validation_dataset is not None:
        validation_dataset_name = 'validation_{}'.format(validation_dataset.split('/')[1])

    number_of_classes = len(dataset_loader.get_classes(train_dataset))

    image_width = 32
    image_height = 32

    learning_rate = 1e-3
    test_size = 0.1
    batch_size = 0.05
    epochs = 20

    images, labels = dataset_loader.load_dataset_images(train_dataset, image_width, image_height, train_dataset_name, load_backup=True, export_dataset=True)
    shuffle(images, labels)

    if validation_dataset is None:
        X, X_test, Y, Y_test = train_test_split(images, labels, test_size=test_size, random_state=17)
        del images, labels
    else:
        X, Y = images, labels
        del images, labels

        X_test, Y_test = dataset_loader.load_dataset_images(validation_dataset, image_width, image_height, validation_dataset_name, load_backup=True, export_dataset=True)
        shuffle(X_test, Y_test)

    Y = to_categorical(Y, number_of_classes)
    Y_test = to_categorical(Y_test, number_of_classes)

    print '\nTrain dataset.....: {}'.format(train_dataset_name)
    if validation_dataset is not None:
        print 'Validation dataset: {}'.format(validation_dataset_name[11:])
    print 'Learning rate.....: {}'.format(learning_rate)
    print 'Image width.......: {}'.format(image_width)
    print 'Image height......: {}'.format(image_height)
    print 'Epochs............: {}'.format(epochs)
    print 'Batch size........: {}'.format(get_batch_size(len(X), batch_size))

    network = cnn.get_network_architecture(image_width, image_height, number_of_classes, learning_rate)

    if (resume_training):
        model = get_model('final_model/final_model.tflearn', network)
    else:
        create_directories()

    backup_network()

    model = tflearn.DNN(
        network,
        tensorboard_verbose=0,
        tensorboard_dir='train_logs/',
        max_checkpoints=3,
        best_checkpoint_path='best_checkpoint/',
        best_val_accuracy=50.0
    )

    try:
        model.fit(
            X, Y,
            validation_set=(X_test, Y_test),
            batch_size=get_batch_size(len(X), batch_size),
            n_epoch=epochs,
            run_id=train_dataset_name,
            show_metric=True,
            shuffle=True
        )
    except KeyboardInterrupt:
        print('Training interruped')

    print('Saving trained model')
    model.save('final_model/final_model.tflearn')


if __name__ == '__main__':
    train()
