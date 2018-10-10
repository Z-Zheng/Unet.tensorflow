import tensorflow as tf

from module.naive_encoder import NaiveEncoder, FlexEncoder
from module.naive_decoder import NaiveDecoder, FlexDecoder


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


if __name__ == '__main__':
    im = tf.ones([1, 512, 512, 3], tf.float32)
    model = DeepUnet(1, False)
    o = model(im)
