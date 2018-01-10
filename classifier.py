from sklearn.externals import joblib
from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import utils
import sys
from melSpec import *

import os

Data_Directory = '/home/ryan/Documents/AlexForAudio_In_Out/'
svm_model = '%s%s' % (Data_Directory, 'svm_model.m')
graph_filename = '%s%s' % (Data_Directory, 'alex_model.pb')

test_wav = '%s%s' % (Data_Directory, 'testfiles/03_a04_anxiety_d.wav')
# img_file_path = '/home/datasets/BabyCry/Crying/2-107351-A'


def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
                            input_map=None,
                            return_elements=None,
                            name='prefix',
                            op_dict=None,
                            producer_op_list=None)
        return graph


def wav_img(wav_file):
    img_path = '%s%s' % (Data_Directory, 'testfiles/')
    filename = wav_file.strip('.wav').split('/')[-1]

    if not os.path.exists(os.path.join(img_path, filename)):
        os.mkdir(os.path.join(img_path, filename))

    speech_spectrums = getSpectrum(wav_file)
    print(len(speech_spectrums))
    for i in range(len(speech_spectrums)):
        image = speech_spectrums[i]
        image = cv2.resize(image, (227,227))

        img_file = os.path.join(img_path, filename, '{}_img.jpg'.format(i))
        plt.imsave(img_file, image)

    return os.path.join(img_path + filename)


def predict(wav_file):

    img_file_path = wav_img(wav_file)

    graph = load_graph(graph_filename)

    input = graph.get_tensor_by_name('prefix/input:0')
    prob = graph.get_tensor_by_name('prefix/test/prob:0')
    fc7 = graph.get_tensor_by_name('prefix/fc7/fc7:0')
    keep_prob = graph.get_tensor_by_name('prefix/Placeholder_1:0')

    images = utils.load_inputs(img_file_path)
    images = np.asarray(images, dtype=np.float32)

    # show the original image
    # utils.vis_square(images)

    with tf.Session(graph=graph) as sess:
        out = sess.run(fc7, feed_dict={
            input: images,
            keep_prob: 1.
        })

    feat = utils.tpm_max(out)

    clf = joblib.load(svm_model)

    [probs] = clf.predict_proba([feat])

    print(clf.predict([feat]))

    return probs


if __name__ == '__main__':

    probs = predict(test_wav)
    # probs = predict('')
    result = ''
    for i in range(len(probs)):
        result = result + '{:.4f} '.format(probs[i])
    print(result)

