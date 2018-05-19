import tensorflow as tf
import os
import cv2
import scipy as sp
import numpy as np
import re


IMAGENET_MEAN = [123.68, 116.779, 103.939]
# IMAGENET_MEAN = [117, 117, 117]

def load_inputs(img_path):

    images = load_images(img_path)

    img = []

    for i in range(len(images)):
#        # # load and preprocess the image
#        img_string = tf.read_file(images[i])
#        img_decoded = tf.image.decode_png(img_string, channels=3)
#        img_resized = tf.image.resize_images(img_decoded, [227, 227])
#        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
#        # RGB -> BGR
#        img_bgr = img_centered[:, :, ::-1]
#        img.append(img_bgr)
#        # print(images[i])

#        im = sp.misc.imread(images[i])
#        im = sp.misc.imresize(im, (227, 227))
#        im = im -IMAGENET_MEAN
#        # im = im[:, :, ::-1]
#        img.append(im)

        im = cv2.imread(images[i])
        im = cv2.resize(im, (227, 227))
        im = im -IMAGENET_MEAN
        # im = im[:, :, ::-1]
        img.append(im)
         
         
    return img


def load_images(image_path):
    images = []
    files = os.listdir(image_path)
    # print(files)
#    permutation = np.random.permutation(len(files))
    for filei in files:
        filename = os.path.join(image_path, filei)
        if os.path.isfile(filename):
            images.append(filename)

    return images


def load_paths(data_path, data_root):
    paths = []
    labels = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            path = line.strip('\n').split(' ')[0]
            path = os.path.join(data_root, path)
            label = line.strip('\n').split(' ')[-1]
            paths.append(path)
            labels.append(label)

    return paths, labels

def load_labels(data_path, data_root):
    labels = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            label = line.strip('\n').split(' ')[-1]
            labels.append(label)

    y = np.array(labels, dtype=np.uint8)

    return y
