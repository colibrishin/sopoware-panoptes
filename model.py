import tensorflow as tf
import tensorflow_addons as tfa
from mobilenetv3small_wm import MobileNetV3SmallSegmentation as MNV3
import datetime

# From "Searching for MobileNetV3" Section 6.1.1
#
# "using standard tensorflow RMSPropOptimizer with 0.9 momentum. 
# We use the initial learning rateof  0.1,  with  batch  size  4096  
# (128  images  per  chip),  and learning  rate  decay  rate  of  0.01  
# every  3  epochs.  We  use dropout of 0.8, and l2 weight decay 1e-5 
# and the same image preprocessing as Inception [42].  Finally we use 
# exponential moving average with decay 0.9999.  All our convolutiona
# l layers use batch-normalization layers with average decay of 0.99."

class MobileNetV3():
    '''
    Please note that this class is not inherit the Tensorflow.keras.Model
    fit(), predict(), evaluate() is overloaded by self.model
    '''

    def __init__(
        self,
        weights_path: str,
        learning_rate: float=5e-05,
        n_classes: int=32, 
        width_multiplier: float=1.25, 
        shape: tuple=(480, 640), 
        avg_pool_kernel: int=11, 
        avg_pool_strides: int=4, 
        axis: int=-1):

        self.n_classes = n_classes
        self.width_multiplier = width_multiplier
        self.avg_pool_kernel = avg_pool_kernel
        self.avg_pool_strides = avg_pool_strides
        self.shape = shape
        self.axis = axis

        self.make_model(self.n_classes, self.width_multiplier, self.shape, self.avg_pool_kernel, self.avg_pool_strides, self.axis)
        self.is_weights_loaded = False
        self.is_model_train_ready = False

        if weights_path != '':
            self.load_weights_to_model(weights_path)

        # Redirect to TF Keras functions
        self.fit = self.model.fit
        self.predict = self.model.predict
        self.evalutae = self.model.evaluate

    def make_model(self, n_classes, width_multiplier, shape, avg_pool_kernel, avg_pool_strides, axis):
        try:
            self.model = MNV3(n_classes=n_classes, 
                width_multiplier=width_multiplier, 
                shape=shape, 
                avg_pool_kernel=avg_pool_kernel, 
                avg_pool_strides=avg_pool_strides, 
                axis=axis)
        except Exception as e:
            raise Exception('Failed to compile model. ', e)
    
    def make_train_ready(self, learning_rate=5e-04):
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=0.9)
        opt = tfa.optimizers.MovingAverage(opt, average_decay=0.9999)

        self.model.compile(
            optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)]
        )
        self.is_model_train_ready = True
    
    def load_weights_to_model(self, weights_path: str):
        try:
            self.model.load_weights(weights_path)
            self.is_weights_loaded = True
        except Exception as e:
            raise Exception("Failed to load weights. ", e)
    
    def change_shape(self, shape:tuple):
        if len(shape) > 2 or len(shape) < 2:
            raise Exception('dimension needs to be 2.')
        self.shape = shape
        self.make_model(n_classes=self.n_classes, 
                width_multiplier=self.width_multiplier, 
                shape=self.shape, 
                avg_pool_kernel=self.avg_pool_kernel,
                avg_pool_strides=self.avg_pool_strides, 
                axis=self.axis)

    def _set_input_shape(self):     
        # Due to using subclassed model, setting the input shape by calling 
        # model.build(input_shape) is impossible. so, compute_output_shape is used.
        # 2021-06-14, Tensorflow 2.4.1
        # [ https://github.com/tensorflow/tensorflow/issues/39906 ]
        self.model.compute_output_shape(input_shape=(None, self.shape[0], self.shape[1], 3))

    def save_model(self, save_dir: str):
        self._set_input_shape()
        self.model.save(save_dir)

def _model_exception_check_weights(model: MobileNetV3):
    if model.is_weights_loaded is False:
        raise Exception('model.load_weight_to_model() need to be called')

def _model_exception_check_if_trainable(model: MobileNetV3):
    if model.is_model_train_ready is False:
        raise Exception('Model is not ready for training.')

def save_model(model: MobileNetV3):
    try:
        _model_exception_check_weights(model)
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = './models/model-' + time
        model.save_model(save_dir)

        print('Model saved at ' + save_dir)
        return save_dir, model
    except Exception as e:
        raise Exception('Failed to save model', e)

def save_model_tflite(
    model: MobileNetV3,
    float16: bool=True):

    _model_exception_check_weights(model)

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = './models/model-' + time
    model.save_model(save_dir)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(save_dir)
    if float16 is True:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(save_dir + '/tflite.model', 'wb') as f:
        f.write(tflite_model)