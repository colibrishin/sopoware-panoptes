import os
import tensorflow as tf
import re
from glob import glob

'''
Dataset structure must be :
    datasets
    |---[some_dataset_name]
        |---[images]
        |   |---[train]
        |   |---[valid]
        |   |---[test]
        |---[labels]
            |---[train]
            |---[valid]
            |---[test]

and, filename of a pair of image and mask must be same.
'''
SHAPE = ()

# <--- Parse Functions begin --->
def parse_png(image_filename: str):
    '''
    Load images and labels. masks are always considered as grayscale.

    image_filename = a filename of image
    returns a dictionary tf dataset, images and masks
    '''

    images = tf.io.read_file(image_filename)
    images = tf.image.decode_png(images, channels=3)
    images = tf.image.convert_image_dtype(images, tf.uint8)

    mask_pat = tf.strings.regex_replace(image_filename, 'images', 'labels')

    masks = tf.io.read_file(mask_pat)
    masks = tf.image.decode_png(masks, channels=1)
    masks = tf.image.convert_image_dtype(masks, tf.uint8)

    return (images, masks)

def parse_jpg(image_filename: str):
    '''
    Load images and labels. label mask input is always considered as grayscale.

    image_filename = a filename of image
    returns a dictionary tf dataset, images and masks
    '''

    images = tf.io.read_file(image_filename)
    images = tf.image.decode_jpeg(images, channels=3)
    images = tf.image.convert_image_dtype(images, tf.uint8)

    mask_pat = tf.strings.regex_replace(image_filename, 'images', 'labels')

    masks = tf.io.read_file(mask_pat)
    masks = tf.image.decode_png(masks, channels=1)
    masks = tf.image.convert_image_dtype(masks, tf.uint8)

    return (images, masks)

# <--- Parse Function end --->

# <--- Map Dataset Functions begin --->
def map_dataset(dir_pat: str):
    dataset = tf.data.Dataset.list_files(dir_pat)
    dataset = dataset.map(parse)

    return dataset

def map_dataset_png(dir_pat: str):
    '''
    Map sets of image and label into tensorflow dataset type.

    dir_pat = the dataset path includes image file extensions by regex expression.
    
    returns a dictionary tf dataset, images and masks 
    '''
    dataset = tf.data.Dataset.list_files(dir_pat)
    dataset = dataset.map(parse_png)
    return dataset

def map_dataset_jpg(dir_pat: str):
    '''
    Map sets of image and label into tensorflow dataset type.

    dir_pat = the dataset path includes image file extensions by regex expression.
    
    returns a dictionary tf dataset, images and masks 
    '''
    dataset = tf.data.Dataset.list_files(dir_pat)
    dataset = dataset.map(parse_jpg)
    return dataset

# <-- Map Dataset Functions end -->
@tf.function
def preprocess_dataset_train(x, y):
    '''
    Preprocess the dataset

    data = A tensorflow dataset
    shape = the output shape of images and masks (height, width)
    preprocess = whether apply the preprocess to image and masks

    returns, preprocessed dictionary tf datasets, images and masks
    '''

    images = tf.image.resize(x, SHAPE)
    masks = tf.image.resize(y, SHAPE)

    
    # You can customize here easily. flip upside down, crop, pad etc.
    if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) >= 5e-1:
        images = tf.image.flip_left_right(images)
        masks = tf.image.flip_left_right(masks)
        
    #if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) >= 5e-1:
    #    images = tf.image.flip_up_down(images)
    #    masks = tf.image.flip_up_down(masks)

    images = tf.cast(images, tf.float32) / 255.0

    return (images, masks)

@tf.function
def preprocess_dataset_valid(x, y):
    '''
    Preprocess the dataset

    data = A tensorflow dataset
    shape = the output shape of images and masks (height, width)
    preprocess = whether apply the preprocess to image and masks

    returns, preprocessed dictionary tf datasets, images and masks
    '''

    images = x
    masks = y

    images = tf.image.resize(images, SHAPE)
    masks = tf.image.resize(masks, SHAPE)

    images = tf.cast(images, tf.float32) / 255.0

    return (images, masks)

def make_batch(
        data,
        shuffle: bool,
        buffer_size: int=250,
        batch_size: int=1,
        preprocess: bool=False,
    ):
    '''
    Create a batch to use dataset in training.

    data = A TF dataset
    buffer_size = Buffer size
    batch_size = batch size
    shuffle = shuffle dataset (True or False)

    returns, a batch dataset
    '''
    if preprocess:
        data = data.map(preprocess_dataset_train)
    else:
        data = data.map(preprocess_dataset_valid)
    if shuffle:
        data = data.shuffle(buffer_size=buffer_size)
    data = data.repeat()
    data = data.batch(batch_size)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return data

def load_dataset(
    path: str,
    file_extension: str,
    shape: tuple=(480, 640)):
    '''
    Load a dataset.

    path = path to the dataset folder
    file_extension = images file extension (masks extension is fixed to png extension)
    preprocess = whether apply preprocess or not. check preprocess_dataset function for more information.
    shape = images, masks shape to resize for model input (height, width)

    returns, a dataset size and a TF dataset consists of images and labels
    '''

    if file_extension == '' or file_extension is None:
        assert 'No file extension given'
    
    path += "*." + file_extension
    
    global SHAPE
    SHAPE = shape

    size = len(glob(path))
    if size == 0:
        assert 'No images/masks found. given path or file extension might be wrong'
    
    if file_extension == 'png':
        data = map_dataset_png(path)
    elif file_extension == 'jpg' or file_extension == 'jpeg':
        data = map_dataset_jpg(path)
    else:
        assert "Unknown extension"

    return size, data

def create_batch_crossvalidation(
        dataset_name: str, 
        file_extension: str,
        shape: tuple=(480, 640),
        batch_size: int=2,
        buffer_size: int=250):
    '''
    Create a batch dataset

    dataset_name = dataset folder name
    file_extension = images file extension
    shape = images, masks shape to resize for model input (height, width)
    batch_size = batch size
    buffer_size = buffer size

    returns, size of training and validation sets each and a dictionary batch dataset with train, valid
    '''
    
    path = "./datasets/" + dataset_name
    train_path = path + "/images/train/"
    valid_path = path + "/images/valid/"

    train_size, train = load_dataset(train_path, file_extension, shape)
    valid_size, valid = load_dataset(valid_path, file_extension, shape)

    train = make_batch(data=train, buffer_size=buffer_size, batch_size=batch_size, shuffle=True, preprocess=True)
    valid = make_batch(data=valid, buffer_size=buffer_size, batch_size=batch_size, shuffle=False, preprocess=False)

    dataset = {'train' : train, 'valid' : valid}

    return train_size, valid_size, dataset
