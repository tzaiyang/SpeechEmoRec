"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com """

import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
import load_emodb

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
train_file = 'Dataset/train_file.txt'
val_file = 'Dataset/val_file.txt'

weights_path_url = 'http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy'

weights_path = weights_path_url.split('/')[-1]
load_emodb.maybe_download(weights_path,weights_path_url,'./')

# Learning params
learning_rate = 0.001
num_epochs = 300 
batch_size = 30
# Network params
dropout_rate = 0.5
num_classes = 7
train_layers = ['fc6','fc7','fc8']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/finetune_alexnet/tensorboard"
checkpoint_path = "/tmp/finetune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
# x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
# y = tf.placeholder(tf.float32, [batch_size, num_classes])
x = tf.placeholder(tf.float32, [None, 227, 227, 3],name='input')
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers,weights_path=weights_path)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

with tf.name_scope("test"):
    prob = tf.nn.softmax(score, name='prob')

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# Start Tensorflow session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
with tf.Session(config=config) as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)
        train_acc = 0.
        train_count = 0
        train_loss = 0
        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            train_op_return,train_acc_value,train_loss_value = sess.run((train_op,accuracy,loss), feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)
            train_loss += train_loss_value
            train_acc += train_acc_value
            train_count += 1
        train_acc /= train_count
        train_loss /= train_count 
        print("{} Training Loss = {:.4f} Accuracy = {:.4f}".format(datetime.now(),train_loss,train_acc))
#        print("{} Training Loss = {:.4f}".format(datetime.now(),train_loss))
#        print("{} Saving checkpoint of model...".format(datetime.now()))


        # Validate the model on the entire validation set
#        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        test_loss = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc,loss_value = sess.run((accuracy,loss), feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_loss += loss_value

            test_acc += acc
            test_count += 1
        test_acc /= test_count
        test_loss /= test_count
#        print("{} Validation Loss = {:.4f} Accuracy = {:.4f}".format(datetime.now(),test_loss,test_acc))
#        print("{} Validation Loss = {:.4f}".format(datetime.now(),test_loss))
#        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

#        print("{} Model checkpoint saved at {}".format(datetime.now(),checkpoint_name))

    graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['test/prob'])

    alexnet_file = 'alexnet.pb'
    tf.train.write_graph(graph, '.', alexnet_file, as_text=False)