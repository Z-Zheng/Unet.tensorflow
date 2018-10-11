import tensorflow as tf

from module.naive_encoder import NaiveEncoder, FlexEncoder
from module.naive_decoder import NaiveDecoder, FlexDecoder, UpsampleType


class Unet(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 use_softmax=True):
        super(Unet, self).__init__()

        self.num_classes = num_classes
        self.use_softmax = use_softmax

        self.encoder = NaiveEncoder()
        self.decoder = NaiveDecoder(num_classes)

    def call(self, inputs, training=None, mask=None):
        c1, c2, c3, c4, c5 = self.encoder(inputs)
        logit = self.decoder((c1, c2, c3, c4, c5))
        ret = {
            'logit': logit
        }
        if not training:
            if self.use_softmax:
                prob = tf.nn.softmax(logit, axis=-1)
            else:
                prob = tf.nn.sigmoid(logit)
            ret.update({
                'prob': prob
            })

        return ret


class MiniUnet(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 use_softmax=True,
                 use_batch_norm=False):
        super(MiniUnet, self).__init__()

        self.num_classes = num_classes
        self.use_softmax = use_softmax

        self.encoder = NaiveEncoder(block_dims=(8, 16, 32, 64, 128), use_batch_norm=use_batch_norm)
        self.decoder = NaiveDecoder(num_classes, block_dims=(64, 32, 16, 8), use_batch_norm=use_batch_norm)

    def call(self, inputs, training=None, mask=None):
        c1, c2, c3, c4, c5 = self.encoder(inputs)
        logit = self.decoder((c1, c2, c3, c4, c5))
        ret = {
            'logit': logit
        }
        if not training:
            if self.use_softmax:
                prob = tf.nn.softmax(logit, axis=-1)
            else:
                prob = tf.nn.sigmoid(logit)
            ret.update({
                'prob': prob
            })

        return ret


class MicroUnet(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 use_softmax=True,
                 use_batch_norm=False):
        super(MicroUnet, self).__init__()

        self.num_classes = num_classes
        self.use_softmax = use_softmax

        self.encoder = NaiveEncoder(block_dims=(32, 64, 128, 256, 512), use_batch_norm=use_batch_norm)
        self.decoder = NaiveDecoder(num_classes, block_dims=(256, 128, 64, 32), use_batch_norm=use_batch_norm)

    def call(self, inputs, training=None, mask=None):
        c1, c2, c3, c4, c5 = self.encoder(inputs)
        logit = self.decoder((c1, c2, c3, c4, c5))
        ret = {
            'logit': logit
        }
        if not training:
            if self.use_softmax:
                prob = tf.nn.softmax(logit, axis=-1)
            else:
                prob = tf.nn.sigmoid(logit)
            ret.update({
                'prob': prob
            })

        return ret


class DeepUnet(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 use_softmax=True,
                 use_batch_norm=False):
        super(DeepUnet, self).__init__()

        self.num_classes = num_classes
        self.use_softmax = use_softmax

        self.encoder = FlexEncoder(block_dims=(8, 16, 32, 64, 128, 256, 512, 1024),
                                   use_batch_norm=use_batch_norm)
        self.decoder = FlexDecoder(num_classes=num_classes,
                                   block_dims=(512, 256, 128, 64, 32, 16, 8),
                                   use_batch_norm=use_batch_norm)

    def call(self, inputs, training=None, mask=None):
        encoder_outs = self.encoder(inputs)
        logit = self.decoder(encoder_outs)
        ret = {
            'logit': logit
        }
        if not training:
            if self.use_softmax:
                prob = tf.nn.softmax(logit, axis=-1)
            else:
                prob = tf.nn.sigmoid(logit)
            ret.update({
                'prob': prob
            })

        return ret


class Unet4Block(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 use_softmax=True,
                 upsample_method=UpsampleType.Deconv,
                 use_batch_norm=False):
        super(Unet4Block, self).__init__()

        self.num_classes = num_classes
        self.use_softmax = use_softmax

        self.encoder = FlexEncoder(block_dims=(64, 128, 256, 512, 1024),
                                   use_batch_norm=use_batch_norm)
        self.decoder = FlexDecoder(num_classes=num_classes,
                                   block_dims=(512, 256, 128, 64),
                                   use_batch_norm=use_batch_norm,
                                   upsample_mehtod=upsample_method)

    def call(self, inputs, training=None, mask=None):
        encoder_outs = self.encoder(inputs)
        logit = self.decoder(encoder_outs)
        ret = {
            'logit': logit
        }
        if not training:
            if self.use_softmax:
                prob = tf.nn.softmax(logit, axis=-1)
            else:
                prob = tf.nn.sigmoid(logit)
            ret.update({
                'prob': prob
            })

        return ret


from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dropout, ReLU, UpSampling2D


class DebugUnet(tf.keras.Model):
    def __init__(self,
                 num_classes=1,
                 use_softmax=False):
        super(DebugUnet, self).__init__()
        self.use_softmax = use_softmax
        self.maxpool2d = MaxPool2D(pool_size=(2, 2))
        self.upsample2d = UpSampling2D(size=(2, 2))
        self.root = tf.keras.Sequential()
        self.root.add(Conv2D(32, 3, 1, 'SAME'))
        self.root.add(ReLU())
        self.root.add(Dropout(1 - 0.8))
        self.root.add(Conv2D(32, 3, 1, 'SAME'))
        self.root.add(ReLU())

        self.downsample1 = tf.keras.Sequential()
        self.downsample1.add(Conv2D(64, 3, 1, 'SAME'))
        self.downsample1.add(ReLU())
        self.downsample1.add(Dropout(1 - 0.8))
        self.downsample1.add(Conv2D(64, 3, 1, 'SAME'))
        self.downsample1.add(ReLU())

        self.downsample2 = tf.keras.Sequential()
        self.downsample2.add(Conv2D(128, 3, 1, 'SAME'))
        self.downsample2.add(ReLU())
        self.downsample2.add(Dropout(1 - 0.8))
        self.downsample2.add(Conv2D(128, 3, 1, 'SAME'))
        self.downsample2.add(ReLU())

        self.middle = tf.keras.Sequential()
        self.middle.add(Conv2D(256, 3, 1, 'SAME'))
        self.middle.add(ReLU())
        self.middle.add(Dropout(1 - 0.8))
        self.middle.add(Conv2D(256, 3, 1, 'SAME'))
        self.middle.add(ReLU())

        self.upsample2_deconv = Conv2DTranspose(256, 2, 2, 'SAME')
        self.upsample2_conv1 = Conv2D(256, 3, 1, 'SAME')
        self.upsample2_conv2 = Conv2D(128, 3, 1, 'SAME')

        self.upsample3_deconv = Conv2DTranspose(128, 2, 2, 'SAME')
        self.upsample3_conv1 = Conv2D(128, 3, 1, 'SAME')
        self.upsample3_conv2 = Conv2D(64, 3, 1, 'SAME')

        self.up_to_root_deconv = Conv2DTranspose(64, 2, 2, 'SAME')
        self.up_to_root_conv1 = Conv2D(64, 3, 1, 'SAME')
        self.up_to_root_conv2 = Conv2D(32, 3, 1, 'SAME')

        self.pred = Conv2D(num_classes, 1, 1)

    def call(self, inputs, training=None, mask=None):
        im_shape = tf.shape(inputs)[1:3]
        c1 = self.root(inputs)
        p1 = self.maxpool2d(c1)
        c2 = self.downsample1(p1)
        p2 = self.maxpool2d(c2)
        c3 = self.downsample2(p2)
        p3 = self.maxpool2d(c3)

        mid = self.middle(p3)

        c4 = self.upsample2_deconv(mid)
        p4 = tf.concat([c4, c3], axis=-1)
        p4_1 = self.upsample2_conv1(p4)
        p4_2 = self.upsample2_conv2(p4_1)

        c5 = self.upsample3_deconv(p4_2)
        p5 = tf.concat([c5, c2], axis=-1)
        p5_1 = self.upsample3_conv1(p5)
        p5_2 = self.upsample3_conv2(p5_1)

        c6 = self.up_to_root_deconv(p5_2)
        p6 = tf.concat([c6, c1], axis=-1)
        p6_1 = self.up_to_root_conv1(p6)
        p6_2 = self.up_to_root_conv2(p6_1)

        f = tf.image.resize_bilinear(p6_2, im_shape, align_corners=True)
        logit = self.pred(f)

        ret = {
            'logit': logit
        }
        if not training:
            if self.use_softmax:
                prob = tf.nn.softmax(logit, axis=-1)
            else:
                prob = tf.nn.sigmoid(logit)
            ret.update({
                'prob': prob
            })
        return ret


if __name__ == '__main__':
    im = tf.ones([1, 512, 512, 3], tf.float32)
    model = DebugUnet(1, )
    o = model(im)
    print(o)
