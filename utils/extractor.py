#Author: khoidd

from os import listdir
import scipy.io.wavfile as wavfile
import pandas as pd


def read_audio(file_name, second=30):
    sr, src = wavfile.read(file_name, mmap=True)

    audio_len = src.shape[0]
    sample_len = sr*second

    if audio_len < sample_len:
        raise "[ERROR] Audio length is not enough!"

    return src[0:sample_len].T


def extract_audio(directory: str, limit):
    cf_song: dict = pd.read_pickle('model/cf_song.pkl')
    # list data file (audio format)
    ls_song_id = set([int(f[:-4]) for f in listdir(directory)])
    if limit > 0 and limit < len(ls_song_id):
        ls_song_id = ls_song_id[:limit]
    # create feature/label matrix
    feature_list = []
    label_list = []
    for song_id in ls_song_id:
        file_name = '%s/%s.wav' % (directory, song_id)
        feature_list.append(read_audio(file_name))
        label_list.append(cf_song.get(song_id))

    feature_data = np.array(feature_list)
    label_data = np.array(label_list)

    return feature_data, label_data


def load_dataset(directory: str, limit=5):
    x_data, y_data = extract_audio(directory, limit)
    print("size of dataset: %i" % (x_data.shape[0]))

    return x_data, y_data
