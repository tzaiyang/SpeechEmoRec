import tensorflow as tf
import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt


IMAGENET_MEAN = [123.68, 116.779, 103.939]
# IMAGENET_MEAN = [117, 117, 117]


def load_inputs(img_path):

    images = load_images(img_path)

    img = []

    for i in range(len(images)):
        # # load and preprocess the image
        # img_string = tf.read_file(images[i])
        # img_decoded = tf.image.decode_png(img_string, channels=3)
        # img_resized = tf.image.resize_images(img_decoded, [227, 227])
        #
        # img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
        #
        # # RGB -> BGR
        # img_bgr = img_centered[:, :, ::-1]

        # print(images[i])

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
    files = sorted(files, key=lambda i: int(re.match(r'(\d+)', i).group()))
    for file in files:
        filename = os.path.join(image_path, file)
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


def tpm_max(features):
    n, d = features.shape
    if n == 3:
        features = np.row_stack((features, features[-1]))

    if n == 2:
        features = np.row_stack((features, features))

    if n == 1:
        features = np.row_stack((features, features, features, features))

    n, d = features.shape

    [a, b], [c, d, e, f] = div(n)

    level1 = np.max(features, axis=0)

    level2_1 = np.max(features[:a], axis=0)
    level2_2 = np.max(features[a:], axis=0)

    level4_1 = np.max(features[:c], axis=0)
    level4_2 = np.max(features[c:a], axis=0)
    level4_3 = np.max(features[a:a+e], axis=0)
    level4_4 = np.max(features[a+e:], axis=0)

    feature = np.concatenate((level1, level2_1, level2_2, level4_1, level4_2, level4_3, level4_4), axis=0)

    return feature


def div(num):
    [a, b] = div2(num)
    [c, d, e, f] = div4(num)

    return [a, b], [c, d, e, f]


def div2(num):
    a = num // 2
    b = num- a

    return [a, b]


def div4(num):
    [a, b] = div2(num)
    [c, d] = div2(a)
    [e, f] = div2(b)

    return [c, d, e, f]


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')
    plt.show()