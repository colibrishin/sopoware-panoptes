import numpy as np

def determine(img: np.array, sidwalk_class: int):
    sidewalk_count = (img == sidewalk_class).sum()
    all_count = img.shape[0] * img.shape[1]
    ratio = sidewalk_count / all_count
    
    trigger = True if ratio > 0.6 else False
    return trigger