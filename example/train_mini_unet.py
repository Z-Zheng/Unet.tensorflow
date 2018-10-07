import tensorflow as tf
from module.unet import MiniUnet
from util import estimator_util
from data import seg_data
from util import learning_rate_util
from module.loss import balance_positive_negative_weight, dice_loss

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import logging

logger = logging.getLogger('tensorflow')
logger.propagate = False
tf.logging.set_verbosity(tf.logging.INFO)


def compute_mean_iou(_, total_cm):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = math_ops.to_float(math_ops.reduce_sum(total_cm, 0))
    sum_over_col = math_ops.to_float(math_ops.reduce_sum(total_cm, 1))
    cm_diag = math_ops.to_float(array_ops.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = math_ops.reduce_sum(
        math_ops.cast(
            math_ops.not_equal(denominator, 0), dtype=dtypes.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = array_ops.where(
        math_ops.greater(denominator, 0), denominator,
        array_ops.ones_like(denominator))
    iou = math_ops.div(cm_diag, denominator)

    # If the number of valid entries is 0 (no classes) we return 0.
    result = array_ops.where(
        math_ops.greater(num_valid_entries, 0),
        math_ops.reduce_sum(iou, name='mean_iou') / num_valid_entries, 0)
    return result


def main():
    # todo: use a config system to manage
    # config
    image_dir = '/media/zhengzhuo/0BE616A90BE616A9/Unet.tensorflow/test_images/images'
    mask_dir = '/media/zhengzhuo/0BE616A90BE616A9/Unet.tensorflow/test_images/masks'
    record_path = '/media/zhengzhuo/0BE616A90BE616A9/Unet.tensorflow/test_images/train.record'
    model_dir = './tmp'

    num_gpus = 0
    num_classes = 2
    num_steps = 10
    eval_per_steps = 1000
    score_threshold = 0.7
    # batch size should be larger than 16 if you use batch normalization
    batch_size = 2
    use_batch_norm = True

    hyper_params = {
        'learning_rate_fn': learning_rate_util.cosine_learning_rate(0.1, num_steps, 0.000001),
        'momentum': 0.9,
        'freeze_prefixes': []
    }

    # 1. define loss and metric
    def loss_fn(predictions, labels, params=None):
        pred_logit = predictions['logit']
        labels = tf.reshape(labels, [-1])
        flat_logit = tf.reshape(pred_logit, [-1, num_classes])
        onehot_labels = tf.one_hot(labels, num_classes)

        bpn_weights = balance_positive_negative_weight(labels, positive_weight=22. / 23.,
                                                       negative_weight=1. / 23.)

        # ce_loss = tf.losses.softmax_cross_entropy(onehot_labels, flat_logit, weights=bpn_weights)
        ce_loss = tf.losses.sigmoid_cross_entropy(onehot_labels, flat_logit, weights=bpn_weights[:, None])
        pred_label = tf.argmax(flat_logit, axis=-1)
        dice_loss_v = dice_loss(pred_label, labels)
        tf.summary.scalar('loss/cross_entropy_loss', ce_loss)
        tf.summary.scalar('loss/dice_loss', dice_loss_v)
        # add metric for training
        pred_class = tf.to_float(tf.sigmoid(flat_logit) > score_threshold)

        miou = tf.metrics.mean_iou(labels, tf.reshape(pred_class, [-1]), num_classes=num_classes)
        miou_v = compute_mean_iou(None, miou[1])
        tf.identity(miou_v, 'train_miou')
        tf.summary.scalar('train/miou', miou_v)

        return ce_loss + dice_loss_v

    def metric_fn(predictions, labels, params=None):
        images = params['inputs']
        MEAN = [[[[122.7717, 115.9465, 102.9801]]]]
        images += MEAN
        pred_prob = predictions['prob']
        pred_class = tf.to_float(pred_prob > score_threshold)

        # add summary for prediction results.
        pred_images = tf.expand_dims(pred_class * 255, axis=-1)
        vis_preds = tf.concat([tf.cast(images, pred_class.dtype), tf.tile(pred_images, multiples=[1, 1, 1, 3])], axis=2)
        vis_preds = tf.cast(vis_preds, tf.uint8)
        tf.summary.image('prediction', vis_preds, max_outputs=10)

        labels = tf.reshape(labels, [-1])
        pred_class = tf.reshape(pred_class, [-1])
        miou = tf.metrics.mean_iou(labels, pred_class, num_classes=num_classes)
        return {
            'eval/miou': miou
        }

    def create_model():
        miniunet = MiniUnet(num_classes=num_classes, use_softmax=False, use_batch_norm=use_batch_norm)

        return miniunet

    # 2. prepare data
    train_dataset = seg_data.SegDataset(image_dir=image_dir,
                                        mask_dir=mask_dir,
                                        record_path=record_path,
                                        crop_size_for_train=(128, 128),
                                        rebuild_record=False)

    test_dataset = seg_data.SegDataset(image_dir=image_dir,
                                       mask_dir=mask_dir,
                                       record_path=record_path,
                                       crop_size_for_train=None,
                                       rebuild_record=False)

    train_input_fn = train_dataset.get_input_fn(batch_size=batch_size, epochs=-1, training=True)
    eval_input_fn = test_dataset.get_input_fn(batch_size=1, epochs=1, training=False)

    # 3. prepare model
    estimator = estimator_util.build_estimator(create_model,
                                               model_dir,
                                               num_gpus=num_gpus,
                                               hpyerparams=hyper_params,
                                               loss_fn=loss_fn,
                                               metric_fn=metric_fn)

    log_hook = tf.train.LoggingTensorHook(
        tensors=['learning_rate', 'train_miou'],
        every_n_iter=10,
    )

    # 4. go on the fly
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=num_steps, hooks=[log_hook])
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=eval_per_steps, start_delay_secs=0, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # estimator.train(train_input_fn, hooks=[log_hook], max_steps=num_steps)
    # estimator.evaluate(eval_input_fn)


if __name__ == '__main__':
    main()
