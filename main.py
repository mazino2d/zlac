from utils import *
import tensorflow as tf

if __name__ == "__main__":
    LIST_POOL_SIZE = [(3, 4), (2, 4), (2, 2), (2, 2), (2, 2), (2, 2)]
    model = gen_model(LIST_POOL_SIZE, rate_dropout=0.05)

    tf.keras.utils.plot_model(model, to_file='arch/version1.png', show_shapes=True, 
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

    model.summary()
