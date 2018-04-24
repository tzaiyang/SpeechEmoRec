# -*- coding: utf-8 -*-

""" AlexNet.
Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def load_data(imagepath_file):
    imagepaths = open(imagepath_file)
    X = []
    Y = []
    for line in imagepaths:
        imagepath = line.strip()
        image = plt.imread(imagepath)
        image_arr = np.array(image)

        X.append(image_arr)
        Y.append(labels_dict[imagepath.split('/')[-1][0]])
        print(image_arr.shape,imagepath.split('/')[-1][0])
    X = np.array(X)
    Y = np.array(Y)
    return X,Y


def alexnet():
    # Building 'AlexNet'
    network = input_data(shape=[None, 227, 227, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 7, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    return network

def training(X,Y):
    # Training
    network = alexnet()
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2)
    model.fit(X, Y, n_epoch=300, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_SpeechEmoRec')

if __name__ == '__main__':
    # 1W-Anger
    # 2L-Boredom
    # 3E-Disgust
    # 4A-Anxiety/Fear
    # 5F-happiness
    # 6T-sadness
    # 7N-neutral
    labels_dict = {'W':[1,0,0,0,0,0,0],
                   'L':[0,1,0,0,0,0,0],
                   'E':[0,0,1,0,0,0,0],
                   'A':[0,0,0,1,0,0,0],
                   'F':[0,0,0,0,1,0,0],
                   'T':[0,0,0,0,0,1,0],
                   'N':[0,0,0,0,0,0,1]}
    imagepath_file = 'Dataset/dcnninname.txt'
    X,Y=load_data(imagepath_file)
    print(X.shape,Y.shape)
    training(X, Y)

