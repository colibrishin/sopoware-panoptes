import tensorflow as tf
import tensorflow_addons as tfa
from dataset import create_batch_crossvalidation
from model import MobileNetV3, model_exception_check_if_trainable
import datetime, os

def train_quick_crossvalidation(
        dataset_name: str,
        file_extension: str,
        n_classes: int=32,
        width_multiplier: float=1.25,
        use_exist_weights: str=None,
        shape: tuple=(480, 640),
        batch_size: int=16, 
        buffer_size: int=250,
        epochs: int=1000,
        learning_rate: float=5e-05,
        save_weights_only: bool=False,
        save_best_only: bool=True,
        verbose: int=1
    ):
    '''
    Pre-set Cross-validation training.
    Returns training history and trained model.

    dataset_name = Dataset folder name
    file_extension = the extension of input image file
    n_classes = the number of class.
    width_multiplier = the multiplier for layers in model
    use_exist_weights = path to exist weights. let it None if you want to create a new weights file.
    shape = the shape of input image, and model.
    batch_size = batch size.
    buffer_size = buffer size (for shuffling the dataset)
    epochs = epochs
    learning_rate = learning rate
    save_weights_only = If this is set to True, weights file will saved without meta data.
    save_best_only = save weights only if validation loss is lower than last saved weights file.
    verbose = verbosity.
    '''

    model = MobileNetV3('')
    model.prepare_train(learning_rate)

    train_size, valid_size, dataset = create_batch_crossvalidation(
        dataset_name, 
        file_extension, 
        shape, 
        batch_size, 
        buffer_size)

    fit(
        model, 
        dataset['train'], 
        dataset['valid'], 
        train_size, 
        valid_size, 
        use_exist_weights, 
        batch_size, 
        epochs, 
        learning_rate, 
        save_weights_only, 
        save_best_only, 
        verbose)

    return history, model

def fit(
    model: MobileNetV3,
    train_dataset: tf.data.Dataset,
    valid_dataset: tf.data.Dataset,
    train_size: int,
    valid_size: int,
    use_exist_weights: str=None,
    batch_size: int=16,
    epochs: int=1000,
    learning_rate: float=5e-05,
    save_weights_only: bool=False,
    save_best_only: bool=True,
    verbose: int=1):

    '''
    model = a TF Keras model.
    train_dataset = a TF Dataset that contains training images and masks.
    valid_dataset = a TF Dataset that contains validation images and masks.
    train_size = the number of the set of images and masks in training dataset.
    valid_size = the number of the set of images and masks in validation dataset.
    use_exist_weights = if it is set, weights in given path will be used in fitting.
    batch_size = batch size.
    epochs = epochs, how many full training steps will be done.
    learning_rate = Learning rate.
    save_weights_only = Save without model metadata.
    save_best_only = In cross-validate, save only if a loss of the epoch is all time low.
    verbose = verbosity (1 or 0)

    returns last result of Sparse Categorical Entropy Loss and top-K-accuracy
    weights file will saved in ./weights/[time], as 'best_weights'
    '''

    try:
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join("logs", time)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        callbacks = [
            tensorboard_callback,
            #tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=verbose),
            tf.keras.callbacks.ModelCheckpoint('./weights/' + time + '/best_weights', verbose=verbose, save_best_only=save_best_only, save_weights_only=save_weights_only)
        ]

        model_exception_check_if_trainable(model)

        if use_exist_weights is not None:
            model.load_weights(use_exist_weights)

        steps_per_epoch = train_size // batch_size
        validation_steps = valid_size // batch_size

        history = model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        return history
    except Exception as e:
        raise Exception('Faild to start training. ', e)

def evaluate_test(
    model: MobileNetV3,
    test_dataset,
    weights_path: str='',
    dataset_size: int=232):

    '''
    model = a TF Keras model.
    test_dataset = the test dataset.
    weights_path = path to weights.
    dataset_size = size of dataset.

    returns last result of Sparse Categorical Entropy Loss and top-K-accuracy
    '''
    try:
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        _model_exception_check_if_trainable(model)
        callbacks = [ tensorboard_callback ]
        if weights_path == '':
            assert 'weights_path is not specified.'
        model.load_weights_to_model(weights_path)

        history = model.evaluate(test, steps=dataset_size)
        return history
    except Exception as e:
        raise Exception('Failed to start evaluation. ', e)