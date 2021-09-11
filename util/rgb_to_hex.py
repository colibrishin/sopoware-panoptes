import imgviz
import numpy as np

def rgb_to_hex(rgb: list):
    code = []
    for i in rgb:
        for j in i:
            if j < 0 or j > 255:
                assert 'Color code out of range'
        code.append('#%02x%02x%02x' % i)
    return code

def construct_label_n_code(labels: list, colors: list):
    labelncode = []
    colors = rgb_to_hex(colors)

    for _, tups in enumerate(zip(labels, colors)):
        labelncode.append(tups)
    
    return labelncode

def labelme_get_colors(n_classes: int, shift: int):
    ret = []
    np_colors = imgviz.label_colormap(n_classes)[shift:]

    for i in np_colors:
        ret.append((i[0], i[1], i[2]))
    return ret