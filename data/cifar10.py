from data import base
import tensorflow as tf

MEAN = [[[122.7717, 115.9465, 102.9801]]]


class Cifar10(base.InputPiepline):
    def __init__(self,
                 record_path,
                 rebuild_record=False,
                 training=True,
                 ):
        super(Cifar10, self).__init__(record_path, rebuild_record)

        self.training = training
        self.crop_size_for_train = None

        self.image_reader = base.ImageReader('jpg', channels=3)

    def get_all_inputs(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        if self.training:
            return list(zip(x_train, y_train))
        else:
            return list(zip(x_test, y_test))

    def encode_feature(self, inputs):
        x, y = inputs

        encoded_im = self.image_reader.encode_image(x)

        feature_dict = {
            'image/encoded': base.bytes_feature(encoded_im),
            'image/format': base.bytes_feature('jpg'.encode()),
            'image/height': base.int64_feature(32),
            'image/width': base.int64_feature(32),
            'label': base.int64_feature(int(y)),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example

    def decode_feature(self, example_proto):
        key_to_features = {
            'image/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature(
                (), tf.string, default_value=b'jpg'),
            'image/height': tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            'image/width': tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            'label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed_features = tf.parse_single_example(example_proto, key_to_features)
        return parsed_features

    def preprocess_for_train(self, inputs):
        im = inputs['image/encoded']
        label = inputs['label']
        # decode
        im = tf.image.decode_image(im, channels=3)
        # set shape
        im = tf.reshape(im, [32, 32, 3])

        # todo: data augmentation

        im = tf.cast(im, tf.float32)
        im = im - tf.constant(MEAN, dtype=tf.float32, shape=[1, 1, 3])
        pad_im = tf.pad(im, paddings=[[2, 2], [2, 2], [0, 0]])
        label = tf.cast(label, tf.int64)
        return pad_im, label

    def preprocess_for_test(self, inputs):
        im = inputs['image/encoded']
        label = inputs['label']
        # decode
        im = tf.image.decode_image(im, channels=3)
        # set shape
        im = tf.reshape(im, [32, 32, 3])


        im = tf.cast(im, tf.float32)
        im = im - tf.constant(MEAN, dtype=tf.float32, shape=[1, 1, 3])
        pad_im = tf.pad(im, paddings=[[2, 2], [2, 2], [0, 0]])
        label = tf.cast(label, tf.int64)
        return pad_im, label
