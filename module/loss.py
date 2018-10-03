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

# todo: 1. focal loss
# todo: 2. OHEM
# todo: 3. Dice loss
# todo: 4. smooth L1
