import glob
import os
import sys
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import ntpath
import tqdm
from math import sqrt
plt.style.use('ggplot')

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def txt_print(text):
    sys.stdout.write(str(text))
    sys.stdout.flush()

def extract_features(parent_dir,sub_dirs,file_ext='*.wav',bands = 20, frames = 41):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            label = ntpath.basename(fn)
            print('----------------File-------------------')
            print(label)
            print('---------------------------------------')
            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[round(start):round(end)]) == window_size):
                    signal = sound_clip[int(start):int(end)]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                    label_base = os.path.splitext(label)[0]
            mfccs.append(mfcc)
            labels.append(label_base)
    features = np.asarray(mfccs).reshape(len(mfccs),bands,frames)
    return np.array(features), np.array(labels, dtype=np.int)




def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
