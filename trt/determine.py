import numpy as np

def determine(img: np.array, sidewalk_class: int):
    img = np.squeeze(img, axis=-1)

    img = img[:, int(img.shape[1]/4):int(img.shape[1]/4 * 3)]
    sidewalk_mask = np.full(img.shape, sidewalk_class).astype(np.uint8)

    sidewalk_count = np.equal(img, sidewalk_mask).sum()
    all_count = img.shape[0] * img.shape[1]
    ratio = sidewalk_count / all_count
    print(ratio)
    
    trigger = True if ratio > 0.5 else False
    return trigger