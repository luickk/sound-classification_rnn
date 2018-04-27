from __future__ import print_function, division
import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from tqdm import tqdm
from optparse import OptionParser
import uuid

from data.data_prog import extract_features, one_hot_encode, txt_print

plt.style.use('ggplot')

def main():

    parser = OptionParser()

    # Only required for labeling - Defines train or label mode
    parser.add_option('-m', '--mode', help='train or pred', dest='mode', default = 'pred')
    # Only required for labeling - Enter Model id here
    parser.add_option('-u', '--uid', help='enter model id here')

    (options, args) = parser.parse_args()

    parent_dir = 'Sound-Data'

    if options.mode == 'train':
        tr_sub_dirs = ['tr']
        tr_features,tr_labels = extract_features(parent_dir,tr_sub_dirs)
        tr_labels = one_hot_encode(tr_labels)

        ts_sub_dirs = ['ts']
        ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs)
        ts_labels = one_hot_encode(ts_labels)

        tf.reset_default_graph()

        learning_rate = 0.008
        training_iters = 1000
        batch_size = 20
        name='RNN_MODEL'


        # Network Parameters
        n_input = 20
        n_steps = 41
        n_hidden = 300
        n_classes = len(tr_labels)
        x = tf.placeholder("float", [None, n_input, n_steps], name="xx")
        y = tf.placeholder("float", [None, n_classes], name="yy")

        weight = tf.Variable(tf.truncated_normal([n_hidden, n_classes]), name="w1")
        bias = tf.Variable(tf.truncated_normal([n_classes]), name="b1")

        def RNN(x, weight, bias):
            stacked_rnn = []
            for iiLyr in range(3):
                stacked_rnn.append(rnn_cell.LSTMCell(n_hidden,state_is_tuple = True))
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
            output, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
            output = tf.transpose(output, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)
            return tf.matmul(last, weight, name="soft") + bias


        logits = RNN(x, weight, bias)
        prediction = tf.nn.softmax(logits, name="soft_pr")

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        # Initializing the variables
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)

            for itr in range(training_iters):
                offset = (itr * batch_size) % (tr_labels.shape[0]+1 - batch_size)
                batch_x = tr_features[offset:(offset + batch_size), :, :]
                batch_y = tr_labels[offset:(offset + batch_size), :]
                session.run(train_op, feed_dict={x: batch_x, y : batch_y})

                loss, acc = session.run([loss_op, accuracy], feed_dict={x: batch_x,
                                                                     y: batch_y})

                print('it',itr,'/',training_iters,'  |','loss:',loss)

            model_uuid = str(uuid.uuid1())
            saver.save(session, './model/'+ model_uuid + '/' + 'model.ckpt', global_step=training_iters+1)
            print('Test accuracy: ',session.run(accuracy, feed_dict={x: ts_features, y: ts_labels}) , 3)

    if options.mode == 'pred':
        model_uuid = options.uid
        ts_sub_dirs = ['example']
        ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs)
        ts_features_pred=ts_features

        g = tf.Graph()
        with g.as_default():
            init_op = tf.global_variables_initializer()
            with tf.Session(graph=g) as session:
                session.run(init_op)
                saver = tf.train.import_meta_graph('./model/'+model_uuid+'/'+'model.ckpt-1001.meta')
                ckpt = tf.train.get_checkpoint_state('./model/'+model_uuid+'/')
                saver.restore(session, ckpt.model_checkpoint_path)
                op_to_restore = g.get_tensor_by_name("soft_pr:0")
                x = g.get_tensor_by_name("xx:0")
                feed={x: ts_features_pred}
                test_pred = session.run(op_to_restore, feed_dict=feed)
                print('----------------------------------------------')
                print('Prediction', test_pred)
                print('----------------------------------------------')

if __name__ == "__main__":
    main()
