import tensorflow as tf


def random_crop(image, mask, crop_size):
    """

    Args:
        image: 3-D tensor of shape [height, width, channel]
        mask: 3-D tensor of shape [height, width, channel]
        crop_size a tuple or list of cropped height and width
    Returns:

    """
    original_shape = tf.shape(image)
    im_static_shape = image.shape.as_list()
    im_channel = im_static_shape[-1]

    mask_static_shape = mask.shape.as_list()
    mask_channel = mask_static_shape[-1]

    blob = tf.concat([image, mask], axis=-1)

    max_offset_height = original_shape[0] - crop_size[0] + 1
    max_offset_width = original_shape[1] - crop_size[1] + 1
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

    cropped = tf.image.crop_to_bounding_box(blob, offset_height, offset_width, crop_size[0], crop_size[1])
    cropped_im = tf.slice(cropped, [0, 0, 0], [-1, -1, im_channel])
    cropped_mask = tf.slice(cropped, [0, 0, im_channel], [-1, -1, mask_channel])

    return cropped_im, cropped_mask


def random_flip_left_right(image, mask):
    original_shape = tf.shape(image)
    mask_shape = tf.shape(mask)

    blob = tf.concat([image, mask], axis=-1)
    fliped = tf.image.random_flip_left_right(blob)

    fliped_im = tf.slice(fliped, [0, 0, 0], [-1, -1, original_shape[-1]])
    fliped_mask = tf.slice(fliped, [0, 0, original_shape[-1]], [-1, -1, mask_shape[-1]])

    return fliped_im, fliped_mask


def random_flip_up_down(image, mask):
    original_shape = tf.shape(image)
    mask_shape = tf.shape(mask)

    blob = tf.concat([image, mask], axis=-1)
    fliped = tf.image.random_flip_up_down(blob)

    fliped_im = tf.slice(fliped, [0, 0, 0], [-1, -1, original_shape[-1]])
    fliped_mask = tf.slice(fliped, [0, 0, original_shape[-1]], [-1, -1, mask_shape[-1]])

    return fliped_im, fliped_mask


def flip_left_right(image, mask=None):
    ret = []
    new_image = image[:, ::-1, :]
    ret.append(new_image)

    if mask is not None:
        new_mask = mask[:, ::-1, :]
        ret.append(new_mask)
    return tuple(ret)


def flip_up_down(image, mask=None):
    ret = []
    new_image = image[::-1, :, :]
    ret.append(new_image)

    if mask is not None:
        new_mask = mask[::-1, :, :]
        ret.append(new_mask)
    return tuple(ret)


def normalize(image, radius=1):
    n_image = 2 * radius / 255. * image - radius
    return n_image


def denormalize(image, radius=1):
    return (image + radius) * 255. / (2 * radius)


def random_adjust_saturation(image,
                             min_delta=0.8,
                             max_delta=1.25,
                             seed=None):
    image = tf.image.random_saturation(image / 255., min_delta, max_delta, seed=seed) * 255.
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


def random_adjust_hue(image,
                      max_delta=0.02,
                      seed=None):
    image = tf.image.random_hue(image / 255., max_delta, seed=seed) * 255.
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


def random_adjust_contrast(image,
                           min_delta=0.8,
                           max_delta=1.25,
                           seed=None):
    image = tf.image.random_contrast(image / 255., min_delta, max_delta, seed=seed) * 255.
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


def random_adjust_brightness(image,
                             max_delta=0.2,
                             seed=None):
    image = tf.image.random_brightness(image / 255, max_delta, seed=seed) * 255.
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    return image


def random_image_scale(image,
                       masks=None,
                       min_scale_ratio=0.5,
                       max_scale_ratio=2.0,
                       seed=None):
    """Scales the image size.
    ref: tensorflow object detection api
    https://github.com/tensorflow/models/blob/42f98218d7b0ee54077d4e07658442bc7ae0e661/research/object_detection/core/preprocessor.py#L767
    Args:
      image: rank 3 float32 tensor contains 1 image -> [height, width, channels].
      masks: (optional) rank 3 float32 tensor containing masks with
        size [height, width, num_masks]. The value is set to None if there are no
        masks.
      min_scale_ratio: minimum scaling ratio.
      max_scale_ratio: maximum scaling ratio.
      seed: random seed.
      preprocess_vars_cache: PreprocessorCache object that records previously
                             performed augmentations. Updated in-place. If this
                             function is called multiple times with the same
                             non-null cache, it will perform deterministically.
    Returns:
      image: image which is the same rank as input image.
      masks: If masks is not none, resized masks which are the same rank as input
        masks will be returned.
    """
    with tf.name_scope('RandomImageScale', values=[image]):
        result = []
        image_shape = tf.shape(image)
        image_height = image_shape[0]
        image_width = image_shape[1]

        size_coef = tf.random_uniform([], min_scale_ratio, max_scale_ratio, tf.float32, seed)

        image_newysize = tf.to_int32(
            tf.multiply(tf.to_float(image_height), size_coef))
        image_newxsize = tf.to_int32(
            tf.multiply(tf.to_float(image_width), size_coef))
        image = tf.image.resize_images(
            image, [image_newysize, image_newxsize], align_corners=True)
        result.append(image)
        if masks is not None:
            masks = tf.image.resize_images(
                masks, [image_newysize, image_newxsize],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                align_corners=True)
            result.append(masks)
        return tuple(result)


def random_distort_color(image, color_ordering=0):
    """Randomly distorts color.
    Randomly distorts color using a combination of brightness, hue, contrast and
    saturation changes. Makes sure the output image is still between 0 and 255.
    Args:
      image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
             with pixel values varying between [0, 255].
      color_ordering: Python int, a type of distortion (valid values: 0, 1).
    Returns:
      image: image which is the same shape as input image.
    Raises:
      ValueError: if color_ordering is not in {0, 1}.
    """
    with tf.name_scope('RandomDistortColor', values=[image]):
        if color_ordering == 0:
            image = random_adjust_brightness(
                image, max_delta=32. / 255.)
            image = random_adjust_saturation(
                image, min_delta=0.5, max_delta=1.5, )
            image = random_adjust_hue(
                image, max_delta=0.2,
            )
            image = random_adjust_contrast(
                image, min_delta=0.5, max_delta=1.5,
            )

        elif color_ordering == 1:
            image = random_adjust_brightness(
                image, max_delta=32. / 255.,
            )
            image = random_adjust_contrast(
                image, min_delta=0.5, max_delta=1.5,
            )
            image = random_adjust_saturation(
                image, min_delta=0.5, max_delta=1.5,
            )
            image = random_adjust_hue(
                image, max_delta=0.2,
            )
        else:
            raise ValueError('color_ordering must be in {0, 1}')
        return image
