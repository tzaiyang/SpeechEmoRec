import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
import utils
import numpy as np

from global_path import Data_Directory

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


if __name__ == '__main__':

    # freeze_graph('/home/liwei/AlexForAudio/snaps/finetune_alexnet/checkpoints/')

    graph_filename = '%s%s' % (Data_Directory, 'alex_model.pb')
    graph = load_graph(graph_filename)

    test_data_path = '%s%s' % (Data_Directory, 'txts/03/test_data.txt')
    train_data_path = '%s%s' % (Data_Directory, 'txts/03/train_data.txt')

    # data_root = '/home/datasets/Bolin_Speech/'
    data_root = '%s%s' % (Data_Directory, 'melSpec_Bolin_Speech/')

    for op in graph.get_operations():
        print(op.name, op.values())

    input = graph.get_tensor_by_name('prefix/input:0')
    prob = graph.get_tensor_by_name('prefix/test/prob:0')
    fc7 = graph.get_tensor_by_name('prefix/fc7/fc7:0')
    keep_prob = graph.get_tensor_by_name('prefix/Placeholder_1:0')

    paths, labels = utils.load_paths(test_data_path, data_root)

    features = []
    for i in range(len(paths)):

        # print(paths[i])
        images = utils.load_inputs(paths[i])
        images = np.asarray(images, dtype=np.float32)

        with tf.Session(graph=graph) as sess:
            # with tf.device('gpu:/0'):
            out = sess.run(fc7, feed_dict={
                input: images,
                keep_prob: 1.
            })

        feat = utils.tpm_max(out)
        features.append(feat)

        print('Gain %d utterance feature' % i)

    features = np.asarray(features, dtype=np.float32)
    print(features.shape)

    test_feature_file = '%s%s' % (Data_Directory, 'test_features.npy')
    train_feature_file = '%s%s' % (Data_Directory, 'train_features.npy')
    np.save(test_feature_file, features)

