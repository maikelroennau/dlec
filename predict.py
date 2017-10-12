import os
import shutil
import numpy as np

from cv2 import imwrite

import cnn
import dataset_loader

import tflearn
from tflearn.data_utils import to_categorical

import tensorflow as tf

from matplotlib import pyplot as plt


def get_model(model_path, number_of_classes):
    network = cnn.get_network_architecture(image_width, image_height, number_of_classes, learning_rate)
    model = tflearn.DNN(network)
    return model

def load_images(images_path, image_width, image_height):
    images = dataset_loader.load_images(images_path, image_width, image_height)
    return images

def load_image(images_path, image_width, image_height, on_demand=False):
    image = dataset_loader.load_image(images_path, image_width, image_height, on_demand)
    return image

def load_classes(classes_path):
    return np.load(classes_path).tolist()

def get_list(array):
    new_list = []
    for item in array:
        new_list.append(item)

    return new_list

def save_image(name, image, path=None):
    if not path == None:
        imwrite('{}/{}'.format(path, name), image)
    else:
        imwrite('{}'.format(name), image)
def generate_confusion_matrix(model, images, classes, number_of_classes):
    print 'Generating confusion matrix'
    data = np.zeros((number_of_classes, number_of_classes))

    for i in xrange(images.shape[0]):
        result = model.predict(images)
        data[np.argmax(classes[i]), result[0].index(max(result[0]))] += 1

    for i in range(len(data)):
    	total = np.sum(data[i])

	for x in range(len(data[0])):
		data[i][x] = data[i][x] / total

    print data

    print '[] Generating graph'
    c = plt.pcolor(data, edgecolors = 'k', linewidths = 4, cmap = 'Blues', vmin = 0.0, vmax = 1.0)
    show_values(c)

def show_values(pc, fmt="%.2f", **kw):
    from itertools import izip

    pc.update_scalarmappable()
    ax = pc.get_axes()
    ax.set_yticks(np.arange(7) + 0.5, minor = False)
    ax.set_xticks(np.arange(7) + 0.5, minor = False)
    ax.set_xticklabels(classes, minor = False)
    ax.set_yticklabels(classes, minor = False)

    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)

        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha = "center", va = "center", color = color, **kw)

def predict(model, images, classes):
    print 'Predicting...'
    predictions = model.predict(images)
    del images
    distribution = {key: 0 for key in classes.values()}

    for prediction_set in predictions:
        print('-' * 42)
        for j, prediction in enumerate(prediction_set):
            print('{:>10}:  {}'.format(classes[j], round(prediction, 4)))

            if prediction_set.tolist().index(max(prediction_set)) == prediction_set.tolist().index(prediction):
                distribution[classes[j]] +=  1

    print('-' * 42)

    print('Predictions distribution:')
    for class_label in distribution.keys():
        print('  {}: {}'.format(class_label, distribution[class_label]))

def predict_on_demand(model, images_path, image_width, image_height, classes):
    print 'Predicting...'
    distribution = {key: 0 for key in classes.values()}

    for image in os.listdir(images_path):
        images = load_image(os.path.join(images_path, image), image_width, image_height, True)
        predictions = model.predict(images)

        del images

        for prediction_set in predictions:
            print('-' * 42)
            for i, prediction in enumerate(prediction_set):
                print('{:>10}:  {}'.format(classes[i], round(prediction, 4)))

                if prediction_set.tolist().index(max(prediction_set)) == prediction_set.tolist().index(prediction):
                    distribution[classes[i]] +=  1

    print('-' * 42)
    print('Predictions distribution:')
    for class_label in distribution.keys():
        print('  {}: {}'.format(class_label, distribution[class_label]))

def visual_evaluation(model, images, classes):
    print 'Predicting...'
    directories = [classes.values()][0]
    for directory in directories:
        try:
            shutil.rmtree(os.path.join('predictions', directory))
            os.makedirs(os.path.join('predictions', directory))
        except OSError:
            os.makedirs(os.path.join('predictions', directory))

    i = 0
    for image in images:
        imwrite('predictions/img.jpg', image)
        prediction = model.predict_label([image])
        os.system('mv {} {}'.format('predictions/img.jpg', os.path.join('predictions', classes[prediction[0][0]], str(i) + '.jpg')))
        i += 1

    print('Images saved to predictions folder')

def evaluate_model(model, images_path, image_width, image_height, number_of_classes, batch_size=0.01):
    images, class_labels = dataset_loader.load_dataset_images(images_path, image_width, image_height)
    labels = to_categorical(class_labels, number_of_classes)

    print('\nEvaluating...')
    print('Evaluation result: {}'.format(round(model.evaluate(images, labels, int((len(images) * batch_size)))[0], 4)))


if __name__ == '__main__':
    model_path = 'final_model/final_model.tflearn'
    images_path = 'Datasets/FER+/PublicTest'
    classes_path = 'data/Collection_classes.npy'

    image_width = 32
    image_height = 32
    learning_rate = 0.001

    classes = load_classes(classes_path)
    number_of_classes = len(classes)

    print('\nGetting model architecture')
    model = get_model(model_path, number_of_classes)

    print('\nLoading model')
    model.load(model_path)

    # images = load_images(images_path, image_width, image_height)

    # generate_confusion_matrix(model, images, classes, number_of_classes)
    # predict(model, images, classes)
    evaluate_model(model, images_path, image_width, image_height, number_of_classes)
    # visual_evaluation(model, images, classes)
    # predict_on_demand(model, images_path, image_width, image_height, classes)
