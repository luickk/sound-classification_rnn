import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from tqdm import tqdm

from data.data_prog import extract_features
from data.data_prog import one_hot_encode

plt.style.use('ggplot')

def main():

    parent_dir = 'Sound-Data'

    tr_sub_dirs = ['tr']
    tr_features,tr_labels = extract_features(parent_dir,tr_sub_dirs)
    tr_labels = one_hot_encode(tr_labels)

    ts_sub_dirs = ['ts']
    ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs)
    ts_labels = one_hot_encode(ts_labels)

    #print('------------------Feature_Set-------------------')
    #print(tr_features)

    tf.reset_default_graph()

    learning_rate = 0.01
    training_iters = 1000
    batch_size = 50
    display_step = 200
    name='RNN_MODEL'


    # Network Parameters
    n_input = 20
    n_steps = 41
    n_hidden = 300
    n_classes = 2

    x = tf.placeholder("float", [None, n_input, n_steps])
    y = tf.placeholder("float", [None, n_classes])

    weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    bias = tf.Variable(tf.random_normal([n_classes]))

    def RNN(x, weight, bias):

        stacked_rnn = []
        for iiLyr in range(3):
            stacked_rnn.append(rnn_cell.LSTMCell(n_hidden,state_is_tuple = True))
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
        output, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        return tf.nn.softmax(tf.matmul(last, weight) + bias)

    prediction = RNN(x, weight, bias)

    # Define loss and optimizer
    loss_f = -tf.reduce_sum(y * tf.log(prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_f)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for itr in tqdm(range(training_iters)):
            offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
            batch_x = tr_features[offset:(offset + batch_size), :, :]
            batch_y = tr_labels[:, :]
            _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y: batch_y})

            if training_iters % display_step == 0:
                # Calculate batch accuracy
                acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
                print ("Iter " + str(training_iters) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

        print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: ts_features, y: ts_labels}) , 3))


if __name__ == "__main__":
    main()
