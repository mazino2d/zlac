#Author: khoidd

import os
import warnings
import pandas as pd
import numpy as np
import librosa
import random as rd
import sklearn


def __read_audio_file(file_name: str) -> (np.ndarray, int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(file_name)

    return y, sr


def __extract_audio_file(file_name: str, label: int, dtime=3):
    y, sr = __read_audio_file(file_name)

    audio_size = y.shape[0]
    sample_size = dtime * sr
    num_of_sample = audio_size // sample_size

    ls_feature = []
    ls_label = []

    if num_of_sample > 0:
        temp = y[0:num_of_sample*sample_size]
        ls_feature.extend(np.split(temp, num_of_sample))
        ls_label.extend([label/320] * num_of_sample)

    return ls_feature, ls_label


def __read_audio_directory(directory: str, label, limit=5, format='wav'):
    ls_song_id = [ f.split('.')[0] 
        for f in os.listdir(directory)
        if f.split('.')[-1] == format
    ][:limit]

    ls_feature = []
    ls_label = []

    for song_id in ls_song_id:
        file_name = '%s/%s.%s' % (directory, song_id, format)
        features, labels = __extract_audio_file(file_name, label)
        ls_feature.extend(features)
        ls_label.extend(labels)

    return ls_feature, ls_label


def load_dataset(limit=5):
    X_128, y_128 = __read_audio_directory('data/scale/128', 128, limit=limit)
    X_192, y_192 = __read_audio_directory('data/scale/192', 192, limit=limit)
    X_320, y_320 = __read_audio_directory('data/origin/320', 320, limit=limit)

    ls_feature = []
    ls_label = []

    for idx in range(len(X_320)):
        rand_num = rd.randint(0,1)
        if rand_num:
            ls_feature.append(np.array((X_128[idx], X_320[idx])))
            ls_label.append(y_128[idx] - y_320[idx])
        else:
            ls_feature.append(np.array((X_320[idx], X_128[idx])))
            ls_label.append(y_320[idx] - y_128[idx])

    for idx in range(len(X_320)):
        rand_num = rd.randint(0,1)
        if rand_num:
            ls_feature.append(np.array((X_192[idx], X_320[idx])))
            ls_label.append(y_192[idx] - y_320[idx])
        else:
            ls_feature.append(np.array((X_320[idx], X_192[idx])))
            ls_label.append(y_320[idx] - y_192[idx])

    feature_data = np.array(ls_feature)
    label_data = np.array(ls_label)

    return feature_data, label_data


def data_split(examples, labels, train_frac, random_state=None):
    '''
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, Y_train, Y_tmp = sklearn.model_selection.train_test_split(
                                        examples, labels, train_size=train_frac, random_state=random_state)

    X_val, X_test, Y_val, Y_test   = sklearn.model_selection.train_test_split(
                                        X_tmp, Y_tmp, train_size=0.5, random_state=random_state)

    return X_train, X_val, X_test,  Y_train, Y_val, Y_test

