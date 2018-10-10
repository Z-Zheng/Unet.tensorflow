import tensorflow as tf
from data import base
import glob
import os
from data.preprocess import random_crop, random_flip_left_right, normalize, random_distort_color
import numpy as np


class SegDataset(base.InputPiepline):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 record_path,
                 image_format='jpg',
                 mask_format='png',
                 crop_size_for_train=(256, 256),
                 rebuild_record=False
                 ):
        super(SegDataset, self).__init__(record_path, rebuild_record)

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_format = image_format
        self.mask_format = mask_format

        self.crop_size_for_train = crop_size_for_train
        self.image_reader = base.ImageReader(image_format=self.image_format, channels=3)
        self.mask_reader = base.ImageReader(image_format=self.mask_format, channels=1)

    def get_all_inputs(self):
        im_path_list = glob.glob(os.path.join(self.image_dir, '*.{}'.format(self.image_format)))
        mask_path_list = [os.path.join(self.mask_dir,
                                       os.path.split(im_path)[-1].replace(self.image_format, self.mask_format)) for
                          im_path in im_path_list]

        ret_list = []
        for im_path, mask_path in zip(im_path_list, mask_path_list):
            ret_list.append((im_path, mask_path))

        return ret_list

    def encode_feature(self, inputs):
        im_path, mask_path = inputs

        image_data = tf.gfile.FastGFile(im_path, 'rb').read()
        mask_data = tf.gfile.FastGFile(mask_path, 'rb').read()

        height, width = self.image_reader.read_image_dims(image_data)

        feature_dict = {
            'image/encoded': base.bytes_feature(image_data),
            'image/format': base.bytes_feature(self.image_format.encode()),
            'image/height': base.int64_feature(height),
            'image/width': base.int64_feature(width),
            'mask/encoded': base.bytes_feature(mask_data),
            'mask/format': base.bytes_feature(self.mask_format.encode()),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example

    def decode_feature(self, example_proto):
        key_to_features = {
            'image/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature(
                (), tf.string, default_value=self.image_format),
            'image/height': tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            'image/width': tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            'mask/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'mask/format': tf.FixedLenFeature(
                (), tf.string, default_value=self.mask_format),
        }
        parsed_features = tf.parse_single_example(example_proto, key_to_features)
        return parsed_features

    def preprocess_for_train(self, inputs):
        encoded_im = inputs['image/encoded']
        encoded_mask = inputs['mask/encoded']
        # decode
        im = tf.image.decode_image(encoded_im, channels=3)
        mask = tf.image.decode_image(encoded_mask, channels=1)
        # set shape
        height = inputs['image/height']
        width = inputs['image/width']
        im = tf.reshape(im, [height, width, 3])
        mask = tf.reshape(mask, [height, width, 1])
        # data augmentation
        im, mask = random_crop(im, mask, crop_size=self.crop_size_for_train)
        im, mask = random_flip_left_right(im, mask)
        im = random_distort_color(im, 0)
        im = normalize(im)
        mask = tf.cast(mask, tf.int64)
        return im, mask

    def preprocess_for_test(self, inputs):
        encoded_im = inputs['image/encoded']
        encoded_mask = inputs['mask/encoded']
        # decode to uint8 tensor
        im = tf.image.decode_image(encoded_im, channels=3)
        mask = tf.image.decode_image(encoded_mask, channels=1)
        # set shape
        height = inputs['image/height']
        width = inputs['image/width']

        im = tf.reshape(im, [height, width, 3])
        mask = tf.reshape(mask, [height, width, 1])

        im = tf.cast(im, tf.float32)
        im = normalize(im)
        mask = tf.cast(mask, tf.int64)
        return im, mask
