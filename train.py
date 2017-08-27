import os
import shutil

import cnn
import dataset_loader

import tflearn
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import shuffle, to_categorical


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

def get_batch_size(number_of_images, percentage):
    if (number_of_images * percentage) > 200:
        return 200
    else:
        return int(number_of_images * percentage)

def train(resume_training=False):
    train_dataset = 'datasets/custom'
    dataset_name = train_dataset.split('/')[1]
    validation_dataset = None

    number_of_classes = len(dataset_loader.get_classes(train_dataset))

    image_width = 48
    image_height = 48

    learning_rate = 0.001
    test_size = 0.1
    batch_size = 0.005
    epochs = 25

    images, labels = dataset_loader.load_dataset_images(train_dataset, image_width, image_height, dataset_name, load_backup=True, export_dataset=True)
    X, X_test, Y, Y_test = train_test_split(images, labels, test_size=test_size, random_state=42)

    del images, labels

    Y = to_categorical(Y, number_of_classes)
    Y_test = to_categorical(Y_test, number_of_classes)

    model = cnn.get_network_architecture(image_width, image_height, number_of_classes, learning_rate)

    if (resume_training):
        model = get_model('final_model/final_model.tflearn', model)
    else:
        create_directories()

    model = tflearn.DNN(
        model,
        tensorboard_verbose=3,
        tensorboard_dir='train_logs/',
        max_checkpoints=3,
        best_checkpoint_path='best_checkpoint/'
    )

    try:
        model.fit(
            X, Y,
            validation_set=(X_test, Y_test),
            batch_size=get_batch_size(len(X), batch_size),
            n_epoch=epochs,
            run_id=dataset_name,
            show_metric=True
        )
    except KeyboardInterrupt:
        print('Training interruped')

    print('Saving trained model')
    model.save('final_model/final_model.tflearn')


if __name__ == '__main__':
    train()
