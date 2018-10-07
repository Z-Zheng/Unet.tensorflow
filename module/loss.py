import tensorflow as tf


def l2_weight_decay(var_list, weight_decay, loss_filter_fn=None):
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    loss_filter_fn = loss_filter_fn or exclude_batch_norm
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in var_list
         if loss_filter_fn(v.name)])
    tf.losses.add_loss(l2_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

    return l2_loss


def balance_positive_negative_weight(labels, positive_weight, negative_weight):
    """

    Args:
        labels: 1-D binary tensor of shape [batch, ], 0 is negative example, 1 is postive example
        positive_weight: scalar
        negative_weight: scalar

    Returns:

    """
    postive_mask = tf.to_float(tf.equal(labels, 1))
    negative_mask = 1. - postive_mask

    positive_weights = tf.to_float(positive_weight) * postive_mask
    negative_weights = tf.to_float(negative_weight) * negative_mask

    ret_weights = positive_weights + negative_weights

    return ret_weights


def _dice_coeff(predictions, labels):
    """

    Args:
        predictions: 1-D binary tensor of shape [N, ]
        labels: 1-D binary tensor of shape [N, ]

    Returns:

    """
    _smooth = 1e-8
    predictions = tf.to_float(predictions)
    labels = tf.to_float(labels)
    intersection = tf.reduce_sum(predictions * labels)

    ret_v = (2. * intersection + _smooth) / (tf.reduce_sum(predictions) + tf.reduce_sum(labels) + _smooth)
    return ret_v


def dice_loss(predictions, labels, loss_collection=tf.GraphKeys.LOSSES):
    """

        Args:
            predictions: 1-D binary tensor of shape [N, ]
            labels: 1-D binary tensor of shape [N, ]

        Returns:

        """
    tf.assert_rank(predictions, rank=1)
    tf.assert_rank(labels, rank=1)
    _loss = 1. - _dice_coeff(predictions, labels)
    tf.losses.add_loss(_loss, loss_collection)
    return _loss


def sigmoid_focal_loss(predictions, onehot_labels, alpha=0.25, gamma=2.0):
    """

    Args:
        predictions: 2-D tensor of shape [N, num_classes]
        onehot_labels: 2-D tensor of shape [N, num_classes]
        alpha:
        gamma:

    Returns:

    """
    if predictions.dtype != tf.float32:
        predictions = tf.to_float(predictions)
    if onehot_labels.dtype != tf.float32:
        onehot_labels = tf.to_float(predictions)
    probs = tf.sigmoid(predictions)
    sigmoid_ce_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot_labels, logits=predictions)

    pt = onehot_labels * probs + (1. - onehot_labels) * (1. - probs)

    gamma_factor = tf.pow(1. - pt, gamma)

    alpha_factor = onehot_labels * alpha + (1. - onehot_labels) * (1. - alpha)

    focal_loss = alpha_factor * gamma_factor * sigmoid_ce_losses
    return focal_loss


def softmax_focal_loss(predictions, onehot_labels, alpha, gamma):
    """

    Args:
        predictions: 2-D tensor of shape [N, num_classes]
        onehot_labels: 2-D tensor of shape [N, num_classes]
        alpha:
        gamma:

    Returns:

    """

    if predictions.dtype != tf.float32:
        predictions = tf.to_float(predictions)
    if onehot_labels.dtype != tf.float32:
        onehot_labels = tf.to_float(predictions)
    probs = tf.nn.softmax(predictions, axis=-1)
    softmax_ce_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=predictions)

    pt = onehot_labels * probs + (1. - onehot_labels) * (1. - probs)

    gamma_factor = tf.pow(1. - pt, gamma)

    alpha_factor = onehot_labels * alpha + (1. - onehot_labels) * (1. - alpha)

    focal_loss = alpha_factor * gamma_factor * softmax_ce_losses
    return focal_loss


def smooth_l1_loss(prediction_tensor, target_tensor, delta, weights):
    """

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.

    """
    return tf.reduce_sum(tf.losses.huber_loss(
        target_tensor,
        prediction_tensor,
        delta=delta,
        weights=tf.expand_dims(weights, axis=2),
        loss_collection=None,
        reduction=tf.losses.Reduction.NONE
    ), axis=2)


# todo: OHEM

if __name__ == '__main__':
    tf.enable_eager_execution()
    labels = tf.constant([1, 0, 0, 0, 0, 1, 0])
    t = balance_positive_negative_weight(labels, positive_weight=22. / 23., negative_weight=1. / 23.)
    print(t)
