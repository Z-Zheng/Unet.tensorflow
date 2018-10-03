import tensorflow as tf
from module.base import Conv2DSame
from tensorflow.keras.layers import MaxPool2D, UpSampling2D, ReLU


class NaiveEncoder(tf.keras.Model):
    def __init__(self,
                 block_dims=(64, 128, 256, 512, 1024)):
        super(NaiveEncoder, self).__init__()
        self.block_dims = block_dims
        self.maxpool2d = MaxPool2D(pool_size=(2, 2))

        self.block1 = tf.keras.Sequential()

        self.block1.add(Conv2DSame(block_dims[0], kernel_size=3))
        self.block1.add(ReLU())
        self.block1.add(Conv2DSame(block_dims[0], kernel_size=3))
        self.block1.add(ReLU())

        self.block2 = tf.keras.Sequential()
        self.block2.add(Conv2DSame(block_dims[1], kernel_size=3))
        self.block2.add(ReLU())
        self.block2.add(Conv2DSame(block_dims[1], kernel_size=3))
        self.block2.add(ReLU())

        self.block3 = tf.keras.Sequential()
        self.block3.add(Conv2DSame(block_dims[2], kernel_size=3))
        self.block3.add(ReLU())
        self.block3.add(Conv2DSame(block_dims[2], kernel_size=3))
        self.block3.add(ReLU())

        self.block4 = tf.keras.Sequential()
        self.block4.add(Conv2DSame(block_dims[3], kernel_size=3))
        self.block4.add(ReLU())
        self.block4.add(Conv2DSame(block_dims[3], kernel_size=3))
        self.block4.add(ReLU())

        self.block5 = tf.keras.Sequential()
        self.block5.add(Conv2DSame(block_dims[4], kernel_size=3))
        self.block5.add(ReLU())
        self.block5.add(Conv2DSame(block_dims[4], kernel_size=3))
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


class NaiveDecoder(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 block_dims=(512, 256, 128, 64)):
        super(NaiveDecoder, self).__init__()
        self.upsample2d = UpSampling2D(size=(2, 2))
        self.block1 = tf.keras.Sequential()
        self.block1.add(Conv2DSame(block_dims[0], kernel_size=3))
        self.block1.add(ReLU())
        self.block1.add(Conv2DSame(block_dims[0], kernel_size=3))
        self.block1.add(ReLU())

        self.block2 = tf.keras.Sequential()
        self.block2.add(Conv2DSame(block_dims[1], kernel_size=3))
        self.block2.add(ReLU())
        self.block2.add(Conv2DSame(block_dims[1], kernel_size=3))
        self.block2.add(ReLU())

        self.block3 = tf.keras.Sequential()
        self.block3.add(Conv2DSame(block_dims[2], kernel_size=3))
        self.block3.add(ReLU())
        self.block3.add(Conv2DSame(block_dims[2], kernel_size=3))
        self.block3.add(ReLU())

        self.block4 = tf.keras.Sequential()
        self.block4.add(Conv2DSame(block_dims[3], kernel_size=3))
        self.block4.add(ReLU())
        self.block4.add(Conv2DSame(block_dims[3], kernel_size=3))
        self.block4.add(ReLU())

        self.cls_conv_pred = Conv2DSame(num_classes, 1)

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

    def get_config(self):
        config = {

        }
        return config


if __name__ == '__main__':
    im = tf.ones([1, 256, 256, 3])
    m = Unet(2)
    c = m(im)
    a = m.trainable_variables
    b = tf.trainable_variables()

    print()
