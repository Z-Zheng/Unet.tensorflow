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
    mask_shape = tf.shape(mask)

    blob = tf.concat([image, mask], axis=-1)

    max_offset_height = original_shape[0] - crop_size[0] + 1
    max_offset_width = original_shape[1] - crop_size[1] + 1
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

    cropped = tf.image.crop_to_bounding_box(blob, offset_height, offset_width, crop_size[0], crop_size[1])
    cropped_im = tf.slice(cropped, [0, 0, 0], [-1, -1, original_shape[-1]])
    cropped_mask = tf.slice(cropped, [0, 0, original_shape[-1]], [-1, -1, mask_shape[-1]])

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


# todo: random scale
