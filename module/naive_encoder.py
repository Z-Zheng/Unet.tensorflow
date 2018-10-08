import tensorflow as tf
from module.base import Conv2DSame
from tensorflow.keras.layers import MaxPool2D, ReLU, BatchNormalization


class NaiveEncoder(tf.keras.Model):
    def __init__(self,
                 block_dims=(64, 128, 256, 512, 1024),
                 use_batch_norm=False):
        super(NaiveEncoder, self).__init__()
        self.block_dims = block_dims
        self.maxpool2d = MaxPool2D(pool_size=(2, 2))

        self.block1 = tf.keras.Sequential()

        self.block1.add(Conv2DSame(block_dims[0], kernel_size=3, use_bias=False))
        if use_batch_norm:
            self.block1.add(BatchNormalization())
        self.block1.add(ReLU())
        self.block1.add(Conv2DSame(block_dims[0], kernel_size=3, use_bias=False))
        if use_batch_norm:
            self.block1.add(BatchNormalization())
        self.block1.add(ReLU())

        self.block2 = tf.keras.Sequential()
        self.block2.add(Conv2DSame(block_dims[1], kernel_size=3, use_bias=False))
        if use_batch_norm:
            self.block2.add(BatchNormalization())
        self.block2.add(ReLU())
        self.block2.add(Conv2DSame(block_dims[1], kernel_size=3, use_bias=False))
        if use_batch_norm:
            self.block2.add(BatchNormalization())
        self.block2.add(ReLU())

        self.block3 = tf.keras.Sequential()
        self.block3.add(Conv2DSame(block_dims[2], kernel_size=3, use_bias=False))
        if use_batch_norm:
            self.block3.add(BatchNormalization())
        self.block3.add(ReLU())
        self.block3.add(Conv2DSame(block_dims[2], kernel_size=3, use_bias=False))
        if use_batch_norm:
            self.block3.add(BatchNormalization())
        self.block3.add(ReLU())

        self.block4 = tf.keras.Sequential()
        self.block4.add(Conv2DSame(block_dims[3], kernel_size=3, use_bias=False))
        if use_batch_norm:
            self.block4.add(BatchNormalization())
        self.block4.add(ReLU())
        self.block4.add(Conv2DSame(block_dims[3], kernel_size=3, use_bias=False))
        if use_batch_norm:
            self.block4.add(BatchNormalization())
        self.block4.add(ReLU())

        self.block5 = tf.keras.Sequential()
        self.block5.add(Conv2DSame(block_dims[4], kernel_size=3, use_bias=False))
        if use_batch_norm:
            self.block5.add(BatchNormalization())
        self.block5.add(ReLU())
        self.block5.add(Conv2DSame(block_dims[4], kernel_size=3, use_bias=False))
        if use_batch_norm:
            self.block5.add(BatchNormalization())
        self.block5.add(ReLU())

    def call(self, inputs, training=None, mask=None):
        x = inputs

        c1 = x = self.block1(x)

        x = self.maxpool2d(x)

        c2 = x = self.block2(x)
        x = self.maxpool2d(x)

        c3 = x = self.block3(x)
        x = self.maxpool2d(x)

        c4 = x = self.block4(x)
        x = self.maxpool2d(x)

        c5 = self.block5(x)
        return c1, c2, c3, c4, c5

    def get_config(self):
        config = {
            'block_dims': self.block_dims
        }
        return config


class FlexEncoder(tf.keras.Model):
    def __init__(self,
                 block_dims=(8, 16, 32, 64, 128, 256, 512, 1024),
                 use_batch_norm=False
                 ):
        super(FlexEncoder, self).__init__()
        self.block_list = []
        self.maxpool2d = MaxPool2D(pool_size=(2, 2))
        # configure per block
        for block_dim in block_dims:
            block_i = tf.keras.Sequential()
            block_i.add(Conv2DSame(block_dim, kernel_size=3, use_bias=False))
            if use_batch_norm:
                block_i.add(BatchNormalization())
            block_i.add(ReLU())
            block_i.add(Conv2DSame(block_dim, kernel_size=3, use_bias=False))
            if use_batch_norm:
                block_i.add(BatchNormalization())
            block_i.add(ReLU())

            self.block_list.append(block_i)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        feat_list = []

        for block_i in self.block_list:
            x = block_i(x)
            feat_list.append(x)
            x = self.maxpool2d(x)

        return feat_list
