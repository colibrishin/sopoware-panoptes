import tensorflow as tf 
import numpy as np

def colorize_mask(mask, labels: np.array):
    mask_shape = mask.shape

    mask = tf.reshape(mask, [-1])
    mask = tf.gather(labels, mask)
    mask = tf.reshape(mask, (mask_shape[0], mask_shape[1], 3))

    return mask

