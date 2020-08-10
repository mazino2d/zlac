#Author: khoidd

from tensorflow.keras import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from kapre.time_frequency import Melspectrogram

def l2_norm(x, axis):
    x_sqr = K.square(x)
    x_sum = K.sum(x_sqr, axis=axis, keepdims=True)
    norm = K.sqrt(x_sum)
    norm = norm + EPS
    return x / norm


def gen_model():

    inpt = Input(
        shape=SHAPE_WAVEFORM,
        name="inpt",
    )
    
    outpt = inpt

    outpt = Melspectrogram(
        n_dft=512,
        n_hop=256,
        power_melgram=2.0,
        trainable_kernel=False,
        trainable_fb=False,
        return_decibel_melgram=True,
        sr=AUDIO_SAMPLE_RATE,
        n_mels=96,
        fmin=0,
        fmax=AUDIO_SAMPLE_RATE // 2,
        name='melgram',
    )(outpt)

    outpt = BatchNormalization(
        axis=AXIS_CHANNEL,
        name='bn_0_freq',
    )(outpt)

    for block_idx in range(num_block):
        outpt = Conv2D(
            filters=LIST_NUM_FEAT_MAP[block_idx],
            kernel_size=LIST_POOL_SIZE[block_idx],
            padding='same',
            name='conv{}'.format(block_idx),
        )(outpt)
        outpt = BatchNormalization(
            axis=AXIS_CHANNEL,
            name='bn{}'.format(block_idx),
        )(outpt)
        outpt = Activation(
            activation='relu',
            name='act{}'.format(block_idx),
        )(outpt)
        outpt = MaxPooling2D(
            pool_size=LIST_POOL_SIZE[block_idx],
            name='pool{}'.format(block_idx),
        )(outpt)

        if block_idx < num_block - 1 and RATE_DROPOUT < 1:
            outpt = Dropout(
                rate=RATE_DROPOUT,
                name='do{}'.format(block_idx),
            )(outpt)

    outpt = Flatten(
        name='flatten',
    )(outpt)

    outpt = Dense(
        units=SIZE_EMBED,
        activation='tanh',
        name="fc",
    )(outpt)

    outpt = Lambda(
        lambda x: l2_norm(x, axis=-1),
        name='l2_norm',
    )(outpt)


    model = Model(inputs=inpt, outputs=outpt, name="ana_audio_model")

    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    return model