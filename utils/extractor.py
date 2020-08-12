#Author: khoidd

import os
import warnings
import pandas as pd
import numpy as np
import librosa


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
        ls_label.extend([label] * num_of_sample)

    return ls_feature, ls_label


def read_audio_directory(directory: str, label, limit=5, format='wav'):
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

    feature_data = np.array(ls_feature)
    label_data = np.array(ls_label)

    return feature_data, label_data
