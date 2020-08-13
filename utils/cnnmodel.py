#Author: khoidd

import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from kapre.time_frequency import Melspectrogram



def gen_model(list_pool_size, rate_dropout=1, axis_channel=1, audio_sample_rate=22050, num_sec=3, is_plot_mode=False):

    shape_wave_form = (axis_channel, audio_sample_rate * num_sec)

    inpt1 = Input(
        shape=shape_wave_form,
        name='inpt_1',
    )

    outpt1 = inpt1
    outpt1 = __gen_melspectrogram(1, outpt1, audio_sample_rate, axis_channel)
    outpt1 = __gen_cnn_layer(1, outpt1, list_pool_size,
                             axis_channel, rate_dropout)
    outpt1 = __gen_fully_connected_layer(1, outpt1)

    inpt2 = Input(
        shape=shape_wave_form,
        name='inpt_2',
    )

    outpt2 = inpt2
    outpt2 = __gen_melspectrogram(2, outpt2, audio_sample_rate, axis_channel)
    outpt2 = __gen_cnn_layer(2, outpt2, list_pool_size,
                             axis_channel, rate_dropout)
    outpt2 = __gen_fully_connected_layer(2, outpt2)

    combined = Subtract()([outpt1, outpt2])

    model = Model(inputs=[inpt1, inpt2], outputs=combined, name='zlac')
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    if is_plot_mode:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, 
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=256)
        
        model.summary()

    return model


def __gen_melspectrogram(index, inpt, audio_sample_rate, batch_norm_axis=1):
    outpt = inpt

    outpt = Melspectrogram(
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
    )(outpt)

    outpt = BatchNormalization(
        axis=batch_norm_axis,
        name='bn_0_freq_%s' % index,
    )(outpt)

    return outpt


def __gen_cnn_layer(index, inpt, list_pool_size, batch_norm_axis=1, rate_dropout=1):
    num_block = len(list_pool_size)
    list_num_feat_map = [32] * num_block

    outpt = inpt

    for block_idx in range(num_block):
        outpt = Conv2D(
            filters=list_num_feat_map[block_idx],
            kernel_size=list_pool_size[block_idx],
            padding='same',
            name='conv{}_{}'.format(block_idx, index),
        )(outpt)
        outpt = BatchNormalization(
            axis=batch_norm_axis,
            name='bn{}_{}'.format(block_idx, index),
        )(outpt)
        outpt = Activation(
            activation='relu',
            name='act{}_{}'.format(block_idx, index),
        )(outpt)
        outpt = MaxPooling2D(
            pool_size=list_pool_size[block_idx],
            name='pool{}_{}'.format(block_idx, index),
        )(outpt)

        if block_idx < num_block - 1 and rate_dropout < 1:
            outpt = Dropout(
                rate=rate_dropout,
                name='do{}_{}'.format(block_idx, index),
            )(outpt)

    return outpt


def __gen_fully_connected_layer(index, inpt):
    outpt = inpt

    outpt = Flatten(
        name='flatten_%s' % (index),
    )(outpt)

    outpt = Dense(
        units=1,
        activation='sigmoid',
        name='fc_%s' % (index),
    )(outpt)

    return outpt
