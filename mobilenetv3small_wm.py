import tensorflow as tf
from tensorflow.keras.layers import Conv2D as TFConv2d, DepthwiseConv2D as TFDepthwiseConv2D, AveragePooling2D as TFAveragePooling2D, GlobalAveragePooling2D as TFGlobalAveragePooling2D
from tensorflow.keras.layers import Dense as TFDense

'''
MobileNetV3 Small - 'Not really Tensorboard Friendly' - implementation
'''

# From Tensorflow
# [https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/conv_blocks.py]
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _identity(x):
    return tf.identity(x)

class Activation():
    def __init__(self, activation_type: str):
        super(Activation, self).__init__()
        self.ret = _identity
        if activation_type == 'RE':
            self.ret = self.relu
        elif activation_type == 'HS':
            self.ret = self.hswish
        elif activation_type == 'hsigmod':
            self.ret = self.hsigmoid
        elif activation_type == 'sigmoid':
            self.ret = self.sigmoid
        elif activation_type == 'softmax':
            self.ret = self.softmax
        elif activation_type == 'NO':
            self.ret = _identity
        
    def relu(self, x):
        return tf.keras.activations.relu(x)
    def relu6(self, x):
        return tf.keras.activations.relu(x, max_value=6)
    def hswish(self, x):
        return x * self.relu6(x + 3.0) / 6.0
    def hsigmod(self, x):
        return tf.keras.activations.hard_sigmoid(x)
    def sigmoid(self, x):
        return tf.keras.activations.sigmoid(x)
    def softmax(self, x):
        return tf.keras.activations.softmax(x)
    def __call__(self, x):
        return self.ret(x)

def Normalization(bn: bool=True, axis=-1):
    if bn:
        return tf.keras.layers.BatchNormalization(
            axis=axis,
            momentum=0.99
        )
    else:
        return _identity

class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel: int,
        strides: int,
        nl: str,
        bn: bool,
        bias: bool=False,
        reg12: float=1e-5,
        axis: int=-1
        ):
        super(ConvBlock, self).__init__()
        self.conv = TFConv2d(
            filters=filters, 
            kernel_size=kernel, 
            strides=strides,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(reg12)
        )
        self.bn = Normalization(bn=bn)
        self.nl = Activation(nl)

    def call(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.nl(x)
        return x

class Bneck(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel: int,
        expansion: int,
        strides: int,
        se: bool,
        nl: str,
        reg12: float=1e-5,
        axis: int=-1
        ):
        super(Bneck, self).__init__()
        self.exp_channels = expansion
        self.out_channels = filters
        self.strides = strides
        
        self.in_block = ConvBlock(
            filters=self.exp_channels,
            kernel=1,
            strides=1,
            bn=True,
            nl=nl,
            reg12=reg12
        )
        self.dw_conv = TFDepthwiseConv2D(
            kernel_size=kernel,
            strides=strides,
            depth_multiplier=1,
            padding='same',
            dilation_rate=(1,1),
            depthwise_regularizer=tf.keras.regularizers.l2(reg12),
            use_bias=False
        )
        self.bn = Normalization()

        if se:
            self.se = SqueezeNExcite()
        else:
            self.se = _identity
        self.nl = Activation(nl)
        self.project = ConvBlock(
            filters=self.out_channels,
            kernel=1,
            strides=1,
            nl='NO',
            bn=True,
            reg12=reg12,
        )
    def build(self, input_shape):
        self.in_channels = int(input_shape[3])
        super().build(input_shape)
    def call(self, input):
        x = self.in_block(input)
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.se(x)
        x = self.nl(x)
        x = self.project(x)
        if self.strides == 1 and self.in_channels == self.out_channels:
            return input + x
        else:
            return x

class SqueezeNExcite(tf.keras.layers.Layer):
    def __init__(
        self,
        axis: int=-1
        ):
        super(SqueezeNExcite, self).__init__()
        self.axis = axis

        self.pool = TFGlobalAveragePooling2D() 

    def build(self, input_shape):
        self.input_channels = input_shape[-1]
        self.fc1 = TFDense(self.input_channels, activation='relu')
        self.fc2 = TFDense(self.input_channels, activation='hard_sigmoid')
        
        super().build(input_shape)

    def call(self, input):
        x = self.pool(input)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.keras.layers.Reshape((1, 1, self.input_channels))(x)
        x = tf.keras.layers.Multiply()([input, x])
        return x

class MobileNetV3Small(tf.keras.Model):
    def __init__(
        self,
        n_classes: int,
        reg12: float=1e-5,
        width_multiplier: float=1.0,
        axis: int=-1
        ):
        super(MobileNetV3Small, self).__init__()
        self.w_m = width_multiplier

        self.conv1 = ConvBlock(kernel=3, filters=_make_divisible(16 * self.w_m, 8) , nl='HS', strides=2, bn=True)

        self.bneck1 = Bneck(kernel=3, expansion=_make_divisible(16 * self.w_m, 8), filters=_make_divisible(16 * self.w_m, 8), se=True, strides=2, nl='RE')

        self.bneck2 = Bneck(kernel=3, expansion=_make_divisible(72 * self.w_m, 8), filters=_make_divisible(24 * self.w_m, 8), se=False, strides=2, nl='RE')
        self.bneck3 = Bneck(kernel=3, expansion=_make_divisible(88 * self.w_m, 8), filters=_make_divisible(24 * self.w_m, 8), se=False, strides=1, nl='RE')

        self.bneck4 = Bneck(kernel=5, expansion=_make_divisible(96 * self.w_m, 8), filters=_make_divisible(40 * self.w_m, 8), se=True, strides=2, nl='HS')

        self.bneck5 = Bneck(kernel=5, expansion=_make_divisible(240 * self.w_m, 8), filters=_make_divisible(40 * self.w_m, 8), se=True, strides=1, nl='HS')
        self.bneck6 = Bneck(kernel=5, expansion=_make_divisible(240 * self.w_m, 8), filters=_make_divisible(40 * self.w_m, 8), se=True, strides=1, nl='HS')
        self.bneck7 = Bneck(kernel=5, expansion=_make_divisible(120 * self.w_m, 8), filters=_make_divisible(48 * self.w_m, 8), se=True, strides=1, nl='HS')
        self.bneck8 = Bneck(kernel=5, expansion=_make_divisible(144 * self.w_m, 8), filters=_make_divisible(48 * self.w_m, 8), se=True, strides=1, nl='HS')



    def call(self, input):
        x2 = self.conv1(input)
        x2 = self.bneck1(x2)
        x2 = x1 = self.bneck2(x2)
        x2 = self.bneck3(x2)
        x2 = self.bneck4(x2)
        x2 = self.bneck5(x2)
        x2 = self.bneck6(x2)
        x2 = self.bneck7(x2)
        x2 = self.bneck8(x2)
        return x1, x2


class SegmentationHead(tf.keras.layers.Layer):
    def __init__(
        self,
        n_classes: int,
        filters: int=128,
        avg_pool_kernel: int=49,
        avg_pool_strides: tuple=(16,20),
        axis=-1
        ):
        super(SegmentationHead, self).__init__()
        self.n_classes = n_classes

        self.out16_b1_conv1 = ConvBlock(filters=filters, strides=1, kernel=1, nl='RE', bn=True)
        self.out16_b2_pool = TFAveragePooling2D(pool_size=avg_pool_kernel, strides=avg_pool_strides)
        self.out16_b2_conv2 = ConvBlock(filters=filters, strides=1, kernel=1, nl='sigmoid', bn=False)
        self.out8_conv1 = ConvBlock(filters=n_classes, kernel=1, strides=1, nl='NO', bn=False)
        self.out16_b2_upsamp = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.out16_conv3 = ConvBlock(filters=n_classes, kernel=1, strides=1, nl='NO', bn=False)
        self.final_upsamp = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation="bilinear")

    def call(self, inputs):
        out8, out16 = inputs

        x1 = self.out16_b1_conv1(out16)
        s = x1.shape

        x2 = self.out16_b2_pool(out16)
        x2 = self.out16_b2_conv2(x2)
        x2 = tf.image.resize(
            x2,
            size=(int(s[1]), int(s[2])),
            method=tf.image.ResizeMethod.BILINEAR
        )

        x = tf.keras.layers.Multiply()([x1, x2])
        x = self.out16_b2_upsamp(x)

        x = self.out16_conv3(x)
        x3 = self.out8_conv1(out8)

        x = tf.keras.layers.Add()([x, x3])

        x = self.final_upsamp(x)
        return x

class MobileNetV3SmallSegmentation(tf.keras.Model):
    
    def __init__(
        self,
        n_classes: int,
        width_multiplier: float=1.0,
        shape: tuple=(480, 640),
        avg_pool_kernel: int=11,
        avg_pool_strides: int=4,
        axis=-1
        ):

        super(MobileNetV3SmallSegmentation, self).__init__()
        self.model = MobileNetV3Small(n_classes, axis=axis, width_multiplier=width_multiplier)
        self.seghead = SegmentationHead(
            n_classes=n_classes,
            avg_pool_kernel=avg_pool_kernel,
            avg_pool_strides=avg_pool_strides,
            axis=axis
        )
    
    
    def call(self, inputs):
        self.seghead_input = self.model(inputs)
        output = self.seghead(self.seghead_input)
        return output



        



