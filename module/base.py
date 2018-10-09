import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
import math


class Conv2DSame(tf.keras.layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=tf.initializers.random_normal(stddev=0.01),
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                 ):
        self.out_channels = filters
        self.k_size = kernel_size
        self.rate = dilation_rate
        self.stride = strides
        padding = 'valid' if self.stride > 1 else 'same'
        super(Conv2DSame, self).__init__(filters,
                                         kernel_size,
                                         strides,
                                         padding,
                                         data_format,
                                         dilation_rate,
                                         activation,
                                         use_bias,
                                         kernel_initializer,
                                         bias_initializer,
                                         kernel_regularizer,
                                         bias_regularizer,
                                         activity_regularizer,
                                         kernel_constraint,
                                         bias_constraint,
                                         **kwargs)

    def call(self, inputs, **kwargs):
        if self.stride == 1:
            return super(Conv2DSame, self).call(inputs)
        kernel_size_effective = self.k_size + (self.k_size - 1) * (self.rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return super(Conv2DSame, self).call(inputs)

    def build(self, input_shape):
        super(Conv2DSame, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        df = conv_utils.normalize_data_format(self.data_format)
        shape = tf.TensorShape(input_shape).as_list()

        kernel_size_effective = self.k_size + (self.k_size - 1) * (self.rate - 1)
        pad_total = kernel_size_effective - 1

        if df == 'channels_first':
            shape[1] = self.out_channels

            shape[2] = math.floor((shape[2] + pad_total - self.rate * (self.k_size - 1) - 1) / self.stride + 1) if \
                shape[2] is not None else None

            shape[3] = math.floor((shape[3] + pad_total - self.rate * (self.k_size - 1) - 1) / self.stride + 1) if \
                shape[3] is not None else None
        else:
            shape[-1] = self.out_channels

            shape[1] = math.floor((shape[1] + pad_total - self.rate * (self.k_size - 1) - 1) / self.stride + 1) if \
                shape[1] is not None else None
            shape[2] = math.floor((shape[2] + pad_total - self.rate * (self.k_size - 1) - 1) / self.stride + 1) if \
                shape[2] is not None else None

        return tf.TensorShape(shape)


class AffineChannel2D(tf.keras.layers.Layer):
    def __init__(self, in_channel, name=None):
        super(AffineChannel2D, self).__init__(name=name)
        self.in_channel = in_channel
        self.bn_s = None
        self.bn_b = None

    def build(self, input_shape):
        self.bn_s = self.add_weight('s',
                                    shape=[self.in_channel],
                                    initializer=tf.constant_initializer(1.0),
                                    trainable=False)
        self.bn_b = self.add_weight('b',
                                    shape=[self.in_channel],
                                    initializer=tf.constant_initializer(0.0),
                                    trainable=False)
        super(AffineChannel2D, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        # [N, H, W, C]
        x = inputs

        ret = x * tf.reshape(self.bn_s, [1, 1, 1, self.in_channel]) + tf.reshape(self.bn_b, [1, 1, 1, self.in_channel])

        return ret

    def compute_output_shape(self, input_shape):
        return input_shape


class Shortcut(tf.keras.Model):
    def __init__(self, out_channel, stride=1, name=None):
        self.out_channel = out_channel
        super(Shortcut, self).__init__(name=name)

        self.conv = Conv2DSame(out_channel, 1, strides=stride, use_bias=False, name='branch1')
        self.bn = AffineChannel2D(out_channel, 'branch1_bn')

    def call(self, inputs, training=None, mask=None):
        x = inputs

        x = self.conv(x)
        x = self.bn(x)
        return x


class Bottleneck(tf.keras.Model):
    def __init__(self,
                 bottleneck_dim,
                 stride=1,
                 stride_1x1=True,
                 in_equal_out=None,
                 name=None):
        """

        Args:
            bottleneck_dim:
            stride:
            stride_1x1: bool, whether place the stride 2 conv on the 1x1 filter
            Use True only for the original MSRA ResNet; use False for C2 and Torch models
            name:
        """
        super(Bottleneck, self).__init__(name=name)
        (str1x1, str3x3) = (stride, 1) if stride_1x1 else (1, stride)
        self.branch2a = Conv2DSame(bottleneck_dim, 1, name='branch2a', use_bias=False, strides=str1x1)
        self.branch2a_bn = AffineChannel2D(bottleneck_dim, name='branch2a_bn')
        self.branch2b = Conv2DSame(bottleneck_dim, 3, name='branch2b', use_bias=False,
                                   strides=str3x3)
        self.branch2b_bn = AffineChannel2D(bottleneck_dim, name='branch2b_bn')
        self.branch2c = Conv2DSame(bottleneck_dim * 4, 1, name='branch2c', use_bias=False)
        self.branch2c_bn = AffineChannel2D(bottleneck_dim * 4, name='branch2c_bn')
        self.relu = tf.keras.layers.ReLU()
        if not in_equal_out:
            self.shortcut_func = Shortcut(bottleneck_dim * 4, stride=stride, name='shortcut')
        else:
            self.shortcut_func = lambda x: x

    def call(self, inputs, training=None, mask=None):
        x = inputs

        x = self.branch2a(x)
        x = self.branch2a_bn(x)
        x = self.relu(x)

        x = self.branch2b(x)
        x = self.branch2b_bn(x)
        x = self.relu(x)

        x = self.branch2c(x)
        x = self.branch2c_bn(x)

        x = x + self.shortcut_func(inputs)

        x = self.relu(x)

        return x


def parse_inputs(inputs):
    pass


if __name__ == '__main__':
    import numpy as np

    tf.enable_eager_execution()
    image = np.arange(25 * 4).reshape((5, 5, 4)).astype(np.float32)
    image = tf.convert_to_tensor(image[None, :, :, :])
    bottle = Bottleneck(1, name='res2_0')
    a = bottle(image)

    print(bottle.weights)

    for var in bottle.weights:
        print(tuple(var.shape))

    print(bottle.weights)
