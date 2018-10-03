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
    cropped_mask = tf.slice(cropped, [0, 0, original_shape[-1] - 1], [-1, -1, mask_shape[-1]])

    new_im_shape = tf.stack([crop_size[0], crop_size[1], original_shape[-1]])
    new_mask_shape = tf.stack([crop_size[0], crop_size[1], mask_shape[-1]])

    # cropped_im = tf.reshape(cropped_im, new_im_shape)
    # cropped_mask = tf.reshape(cropped_mask, new_mask_shape)

    return cropped_im, cropped_mask
