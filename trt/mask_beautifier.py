import numpy as np

def colorize_mask(mask, labels: np.array):
    mask_shape = mask.shape

    ret = np.take(labels, mask, axis=0)
    ret = np.reshape(ret, (mask_shape[0], mask_shape[1], labels.shape[-1]))

    return ret 

