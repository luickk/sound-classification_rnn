from __future__ import print_function, division
import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from tqdm import tqdm

from data.data_prog import extract_features, one_hot_encode, txt_print

plt.style.use('ggplot')


def main():
    parent_dir = 'Sound-Data'
    ts_sub_dirs = ['example']
    ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs)
    ts_features_pred=ts_features

    g = tf.Graph()
    with g.as_default():
        init_op = tf.global_variables_initializer()
        with tf.Session(graph=g) as session:
            session.run(init_op)
            saver = tf.train.import_meta_graph('model/model.ckpt-1001.meta')
            ckpt = tf.train.get_checkpoint_state('model/')
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
