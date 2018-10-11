import tensorflow as tf
from module.base import Conv2DSame
from tensorflow.keras.layers import UpSampling2D, ReLU, BatchNormalization, Conv2DTranspose
import enum


class UpsampleType(enum.Enum):
    Bilinear = 0
    Deconv = 1


class NaiveDecoder(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 block_dims=(512, 256, 128, 64),
                 use_batch_norm=False):
        super(NaiveDecoder, self).__init__()
        self.upsample2d = UpSampling2D(size=(2, 2))
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

        self.cls_conv_pred = Conv2DSame(num_classes, 1, use_bias=True)

    def call(self, inputs, training=None, mask=None):
        c1, c2, c3, c4, c5 = inputs

        p5 = self.upsample2d(c5)
        x = tf.concat([c4, p5], axis=-1)
        c6 = self.block1(x)

        p6 = self.upsample2d(c6)
        x = tf.concat([c3, p6], axis=-1)
        c7 = self.block2(x)

        p7 = self.upsample2d(c7)
        x = tf.concat([c2, p7], axis=-1)
        c8 = self.block3(x)

        p8 = self.upsample2d(c8)
        x = tf.concat([c1, p8], axis=-1)
        c9 = self.block4(x)

        logit = self.cls_conv_pred(c9)

        return logit

    def get_config(self):
        config = {
            'block_dims': self.block_dims
        }
        return config


class FlexDecoder(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 block_dims=(512, 256, 128, 64, 32, 16, 8),
                 upsample_mehtod=UpsampleType.Bilinear,
                 use_batch_norm=False):
        super(FlexDecoder, self).__init__()
        self.cls_conv_pred = Conv2DSame(num_classes, 1, use_bias=True)
        if upsample_mehtod == UpsampleType.Bilinear:
            self.upsample2d = UpSampling2D(size=(2, 2))
        elif upsample_mehtod == UpsampleType.Deconv:
            self.upsample2d = [Conv2DTranspose(out_dim, kernel_size=2, strides=2, padding='same') for out_dim in
                               block_dims]
        else:
            raise ValueError('upsample_method is only support Bilinear mode.')
        self.block_list = []

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
        feat_list = inputs

        feat_list.reverse()
        x_i = feat_list[0]
        out_final = None
        for i, c_i in enumerate(feat_list[:-1]):
            x_i_before = feat_list[i + 1]

            if isinstance(self.upsample2d, list):
                p_i = self.upsample2d[i](x_i)
            else:
                p_i = self.upsample2d(x_i)

            concat_i = tf.concat([x_i_before, p_i], axis=-1)
            out_i = self.block_list[i](concat_i)

            x_i = out_i

            out_final = out_i

        out_final = self.cls_conv_pred(out_final)

        return out_final
