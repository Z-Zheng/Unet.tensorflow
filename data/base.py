from abc import abstractmethod
import tensorflow as tf
import logging
from tensorflow.contrib.data import shuffle_and_repeat, map_and_batch, AUTOTUNE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InputPiepline():
    def __init__(self,
                 record_path,
                 rebuild_record=False):
        self.record_path = record_path
        self.rebuild_record = rebuild_record

    def build_record(self):
        with tf.python_io.TFRecordWriter(self.record_path) as tfw:
            all_inputs = self.get_all_inputs()
            total = len(all_inputs)

            for idx, inputs in enumerate(all_inputs):
                example = self.encode_feature(inputs)
                tfw.write(example.SerializeToString())
                logger.info('[Building Record] #examples {}/{}'.format(idx + 1, total))

    def get_input_fn(self,
                     batch_size,
                     epochs=-1,
                     num_parallel_calls=4,
                     training=False):
        dataset = tf.data.TFRecordDataset(self.record_path)
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.map(self.decode_feature, num_parallel_calls=num_parallel_calls)
        if training:
            dataset = dataset.apply(map_and_batch(self.preprocess_for_train, batch_size=batch_size,
                                                  num_parallel_calls=num_parallel_calls))
            dataset = dataset.apply(shuffle_and_repeat(AUTOTUNE, epochs))
        else:
            dataset = dataset.apply(map_and_batch(self.preprocess_for_test, batch_size=batch_size,
                                                  num_parallel_calls=num_parallel_calls))
        return dataset

    @abstractmethod
    def preprocess_for_train(self, inputs):
        return NotImplementedError

    @abstractmethod
    def preprocess_for_test(self, inputs):
        return NotImplementedError

    @abstractmethod
    def get_all_inputs(self):
        return NotImplementedError

    @abstractmethod
    def encode_feature(self, inputs):
        """

        Args:
            inputs: a dict of input

        Returns:

        """
        return NotImplementedError

    @abstractmethod
    def decode_feature(self, example_proto):
        """

        Args:
            example_proto:

        Returns:

        """
        return NotImplementedError


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, image_format='jpeg', channels=3):
        """Class constructor.
        Args:
          image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
          channels: Image channels.
        """
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if self._image_format in ('jpeg', 'jpg'):
                self._decode = tf.image.decode_jpeg(self._decode_data,
                                                    channels=channels)
            elif self._image_format == 'png':
                self._decode = tf.image.decode_png(self._decode_data,
                                                   channels=channels)

    def read_image_dims(self, image_data):
        """Reads the image dimensions.
        Args:
          image_data: string of image data.
        Returns:
          image_height and image_width.
        """
        image = self.decode_image(image_data)
        return image.shape[:2]

    def decode_image(self, image_data):
        """Decodes the image data string.
        Args:
          image_data: string of image data.
        Returns:
          Decoded image data.
        Raises:
          ValueError: Value of image channels not supported.
        """
        image = self._session.run(self._decode,
                                  feed_dict={self._decode_data: image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')

        return image


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
