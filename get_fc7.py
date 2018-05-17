import os
import tensorflow as tf
import utils
import numpy as np
import path

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


def get_fc7(graph_filename,load_filename):
    graph = load_graph(graph_filename)

    #for op in graph.get_operations():
    #    print(op.name, op.values())

    input = graph.get_tensor_by_name('prefix/input:0')
    prob = graph.get_tensor_by_name('prefix/test/prob:0')
    fc7 = graph.get_tensor_by_name('prefix/fc7/fc7:0')
    keep_prob = graph.get_tensor_by_name('prefix/Placeholder_1:0')

    paths, labels = utils.load_paths(load_filename, './')
    features = []
    with tf.Session(graph=graph) as sess:
    # with tf.device('gpu:/0'):
        for i in range(len(paths)):
            path.DataDir.percent_bar(i+1,len(paths))
            #print(paths[i])
            images = utils.load_inputs(paths[i])
            #print(images)
            images = np.asarray(images, dtype=np.float32)
            out = sess.run(fc7, feed_dict={
                input: images,
                keep_prob: 1.
            })
            features.append(out)
            #print('Gain %d utterance feature' % i)

    return features,labels

if __name__ == '__main__':
    print('test')

