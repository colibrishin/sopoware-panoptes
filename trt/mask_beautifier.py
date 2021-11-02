import numpy as np

def colorize_mask(mask, palette):
    mask_shape = mask.shape

    ret = np.take(palette, mask, axis=0)
    ret = np.reshape(ret, (mask_shape[0], mask_shape[1], 3)).astype('uint8')

    return ret 

