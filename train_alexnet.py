import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
import matplotlib.pyplot as plt


# Path to the textfiles for the trainings and validation set
Data_Directory = '/home/tzaiyang/Documents/AlexForAudio_In_Out/'

train_file = '%s%s' % (Data_Directory, 'txts/03/train.txt')
val_file = '%s%s' % (Data_Directory, 'txts/03/test.txt')
data_root = '%s%s' % (Data_Directory, 'Bolin_Speech')


# Learning params
learning_rate = 0.001
num_epochs = 30
batch_size = 64

# Network params
dropout_rate = 0.5
num_classes = 7
train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints

filewriter_path = '%s%s' % (Data_Directory, 'snaps/finetune_alexnet/tensorboard')
checkpoint_path = '%s%s' % (Data_Directory, 'snaps/finetune_alexnet/checkpoints')

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 data_root,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  data_root,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
# x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3], name='input')
# y = tf.placeholder(tf.float32, [batch_size, num_classes])
x = tf.placeholder(tf.float32, [None, 227, 227, 3], name='input')
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
init_weight_path = '%s%s' % (Data_Directory, 'bvlc_alexnet.npy')
model = AlexNet(x, keep_prob, num_classes, ['fc8'],init_weight_path)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
print(len(tf.trainable_variables()))
print([v.name for v in tf.trainable_variables()])
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
print(var_list)

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # grads2 = [(tf.where(tf.is_nan(grad), tf.zeros(grad.shape), grad), var) for grad, var in gradients]

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
with tf.Session() as sess:
    with tf.device('/gpu:1'):
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)

        print(len(tf.trainable_variables()))
        print([v.name for v in tf.trainable_variables()])

        names = [op.name for op in sess.graph.get_operations()]
        print(names)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

        # Loop over number of epochs
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch+1))

            # Initialize iterator with the training dataset
            sess.run(training_init_op)

            for step in range(train_batches_per_epoch):

                # get next batch of data
                img_batch, label_batch = sess.run(next_batch)

                # And run the training op
                sess.run(train_op, feed_dict={x: img_batch,
                                              y: label_batch,
                                              keep_prob: dropout_rate})

                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            keep_prob: 1.})

                    writer.add_summary(s, epoch*train_batches_per_epoch + step)

            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.now()))
            sess.run(validation_init_op)
            test_acc = 0.
            test_count = 0
            for _ in range(val_batches_per_epoch):

                img_batch, label_batch = sess.run(next_batch)

                acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: 1.})
                test_acc += acc
                test_count += 1
            test_acc /= test_count
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                           test_acc))
            print("{} Saving checkpoint of model...".format(datetime.now()))

            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))

        graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['test/prob'])

        alex_model_file = '%s%s' % (Data_Directory,'alex_model.pb')
        tf.train.write_graph(graph, '.', alex_model_file, as_text=False)
