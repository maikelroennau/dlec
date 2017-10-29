import copy
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Process

import numpy as np
import tensorflow as tf
import tflearn
from cv2 import (CASCADE_SCALE_IMAGE, COLOR_BGR2GRAY, CascadeClassifier,
                 VideoCapture, cvtColor, destroyAllWindows, imshow, imwrite,
                 rectangle, resize, waitKey)
from matplotlib import pyplot as plt
from tflearn.data_utils import to_categorical

import dataset_loader

if (os.path.isfile('final_model/cnn.py')):
    from final_model import cnn
else:
    import cnn


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
    c = plt.pcolor(data, edgecolors='k', linewidths=4, cmap='Blues', vmin=0.0, vmax=1.0)
    show_values(c)

def show_values(pc, fmt='%.2f', **kw):
    from itertools import izip

    pc.update_scalarmappable()
    ax = pc.get_axes()
    ax.set_yticks(np.arange(7) + 0.5, minor=False)
    ax.set_xticks(np.arange(7) + 0.5, minor=False)
    ax.set_xticklabels(classes, minor=False)
    ax.set_yticklabels(classes, minor=False)

    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)

        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha='center',
                    va='center', color=color, **kw)

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
                distribution[classes[j]] += 1

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
                    distribution[classes[i]] += 1

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
    print('Evaluation result: {}'.format(round(model.evaluate(
        images, labels, int((len(images) * batch_size)))[0], 4)))

def camera_prediction(model):
    frequency = 0.25  # seconds
    instant = -1

    faceCascade = CascadeClassifier('haarcascade_frontalface_alt.xml')
    capture = VideoCapture(0)

    while True:
        ret, frame = capture.read()
        image = copy.deepcopy(frame)

        gray = cvtColor(frame, COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=CASCADE_SCALE_IMAGE
        )

        captured_face = None

        for (x, y, w, h) in faces:
            rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            captured_face = gray[y: y + h, x: x + w]

        imshow('Face detector', frame)

        if captured_face is not None:
            captured_face = resize(captured_face, (32, 32))

            instant = classify_emotion(
                model, captured_face, instant, frequency)
            captured_face = None

        if waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    destroyAllWindows()

def classify_emotion(model, frame, instant, frequency):
    if instant < 0 or time.time() - instant >= 60 * frequency:

        instant = time.time()

        process = Process(run_prediction(model, frame))
        process.start()

        return instant
    return instant

def run_prediction(model, frame):
    if os.path.exists('/tmp/images'):
        shutil.rmtree('/tmp/images')

    os.makedirs('/tmp/images')

    imwrite('/tmp/images/to_classify.jpg', frame)

    print ''
    predict_on_demand(model, '/tmp/images', image_width, image_height, classes)


if __name__ == '__main__':
    model_path = 'final_model/final_model.tflearn'

    if len(sys.argv) > 2:
        images_path = sys.argv[1]
        task = int(sys.argv[2])
    else:
        if not os.path.isdir(sys.argv[1]): 
            print 'Inform images path and task type'
            print '  1 - Evalutate model (classes path)'
            print '  2 - Predict on demand (images path)'
            print '  3 - Predict in memory (images path)'
            print '  4 - Visual evaluation (images path)'
            exit(0)
        else:
            images_path = sys.argv[1]
            task = 1

    classes_path = 'data/classes_classes.npy'
    for file in os.listdir('data'):
        if not file.startswith('validation_') and file[-12:].endswith('_classes.npy'):
            classes_path = 'data/{}'.format(file)

    image_width = 32
    image_height = 32
    learning_rate = 1e-3

    classes = load_classes(classes_path)
    number_of_classes = len(classes)

    print('\nGetting model architecture')
    model = get_model(model_path, number_of_classes)

    print('\nLoading model')
    model.load(model_path)

    if task == 1:
        evaluate_model(model, images_path, image_width,
                       image_height, number_of_classes)
    elif task == 2:
        predict_on_demand(model, images_path, image_width,
                          image_height, classes)
    elif task == 3:
        images = load_images(images_path, image_width, image_height)
        predict(model, images, classes)
    elif task == 4:
        images = load_images(images_path, image_width, image_height)
        visual_evaluation(model, images, classes)
    elif task == 5:
        camera_prediction(model)
