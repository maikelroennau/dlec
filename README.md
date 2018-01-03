# DLEC - Deep Learning Emotion Classifier

## Description

DLEC is a convolutional neural network that classifies images of human facial expressions in emotions. 
<!-- -->
It can be used as a framework for developing general purpose deep learning classifiers.

## Motivation

This is a final year project to achieve the Bachelor degree in Computer Science Lutheran University of Brazil.

## Objectives

Accomplish researches on deep learning and develop a model capable to classify human emotions in images.

## How to use

### Configuring the framework

`train.py`

- Set the `train_dataset` path
- Set the `validation_dataset` path (if empty, the train dataset will be slited into train and validation)
- Set `image_width` and `image_height`
- Set `learning_rate`, `test_size` (only used when `validation_dataset` is `None`), `batch_size` and `epochs`

`cnn.py`

- Define the architecture of the CNN in the function `get_network_architecture`
- Use the inline documentation as reference

`predict.py`

- Set the same `image_width` and `image_height` as in `train.py`

### Training

- Make sure the images exist under the paths set in `train_dataset` and `validation_dataset`
- The images of each class should be in its own directory. See the configuration below:

>          Datasets
>          /      \
>       Train   Validation
>        /           \
>     Class 1,      Class 1,
>     Class 2,      Class 2,
>       ...,          ...,
>     Class n,      Class n,

- Run the training with the command `python train.py`
- The training status will be outputed during training

### Predicting

- Run `python predict.py` to get help on the available prediction methods

<!-- -->

- Examples:
    - `python predict.py 1 /Datasets/Test/` (evaluates the model)
    - `python predict.py 6` (real time predictions using the embedded camera)
    - `python predict.py <TASK> <PATH>` (general case)

<!-- -->

- Option 1 does not require the task specification. The command `python predict.py <PATH>` will run into task 1
- Option 6 does not need any path, since it will use the embedded camera as source

### Technical notes

- The code is runnable both in Python 2 and Python 3
- The code also works on Windows (it was tested on Windows 10). Keep in mind adjustments might be required for your environment.

## Author

**Maikel Maciel Rönnau**  
*Computer Scientist  
maikel.ronnau@gmail.com  
[Linkedin](https://br.linkedin.com/in/maikelronnau) - [GitHub](https://github.com/maikelronnau)*
