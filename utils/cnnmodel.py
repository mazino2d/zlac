#Author: khoidd

from tensorflow.keras import Model
import tensorflow.keras.layers as L
import kapre.time_frequency as A
from tensorflow.keras.utils import plot_model
import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def gen_model(list_pool_size, rate_dropout=1, axis_channel=1, audio_sample_rate=22050, num_sec=3, is_plot_mode=False):

    shape_wave_form = (axis_channel, audio_sample_rate * num_sec)

    inpt = L.Input(
        shape=(2, *shape_wave_form),
        name='inpt',
    )

    oupt = inpt

    oupt = __gen_melspectrogram(oupt, audio_sample_rate, axis_channel)
    oupt = __gen_cnn_layer(oupt, list_pool_size, axis_channel, rate_dropout)
    oupt = __gen_fully_connected_layer(oupt)
    oupt = L.Subtract(oupt)

    model = Model(inputs=inpt, outputs=oupt, name='zlac')
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    if is_plot_mode:
        plot_model(model, to_file='model.png', show_shapes=True,
                   show_layer_names=True, rankdir='TB', expand_nested=False, dpi=256)

        model.summary()

    return model


def __gen_melspectrogram(inpt, audio_sample_rate, batch_norm_axis=1, index=0):
    outpt = inpt

    outpt = L.TimeDistributed(A.Melspectrogram(
        n_dft=512,
        n_hop=256,
        power_melgram=2.0,
        trainable_kernel=False,
        trainable_fb=False,
        return_decibel_melgram=True,
        sr=audio_sample_rate,
        n_mels=96,
        fmin=0,
        fmax=audio_sample_rate // 2,
        name='melgram_%s' % index,
    ))(outpt)

    outpt = L.TimeDistributed(L.BatchNormalization(
        axis=batch_norm_axis,
        name='bn_0_freq_%s' % index,
    ))(outpt)

    return outpt


def __gen_cnn_layer(inpt, list_pool_size, batch_norm_axis=1, rate_dropout=1, index=0):
    num_block = len(list_pool_size)
    list_num_feat_map = [32] * num_block

    outpt = inpt

    for block_idx in range(num_block):
        outpt = L.TimeDistributed(L.Conv2D(
            filters=list_num_feat_map[block_idx],
            kernel_size=list_pool_size[block_idx],
            padding='same',
            name='conv{}_{}'.format(block_idx, index),
        ))(outpt)
        outpt = L.TimeDistributed(L.BatchNormalization(
            axis=batch_norm_axis,
            name='bn{}_{}'.format(block_idx, index),
        ))(outpt)
        outpt = L.TimeDistributed(L.Activation(
            activation='relu',
            name='act{}_{}'.format(block_idx, index),
        ))(outpt)
        outpt = L.TimeDistributed(L.MaxPooling2D(
            pool_size=list_pool_size[block_idx],
            name='pool{}_{}'.format(block_idx, index),
        ))(outpt)

        if block_idx < num_block - 1 and rate_dropout < 1:
            outpt = L.Dropout(
                rate=rate_dropout,
                name='do{}_{}'.format(block_idx, index),
            )(outpt)

    return outpt


def __gen_fully_connected_layer(inpt, index=0):
    outpt = inpt

    outpt = L.TimeDistributed(L.Flatten(
        name='flatten_%s' % (index),
    ))(outpt)

    outpt = L.TimeDistributed(L.Dense(
        units=1,
        activation='sigmoid',
        name='fc_%s' % (index),
    ))(outpt)

    return outpt
