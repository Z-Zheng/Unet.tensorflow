import tensorflow as tf
from module.unet import DeepUnet,Unet4Block
from util import estimator_util
from data import seg_data
from util import learning_rate_util
from module.loss import balance_positive_negative_weight, dice_loss
from module.metric import positive_iou, compute_mean_iou, compute_positive_iou
from module.metric import negative_iou, compute_negative_iou
import logging
from tensorboard.summary import pr_curve
from data.preprocess import denormalize

logger = logging.getLogger('tensorflow')
logger.propagate = False
tf.logging.set_verbosity(tf.logging.INFO)


def main():
    # todo: use a config system to manage
    # config
    image_dir = '/home/zf/zz/DATA/mass_roads/train/sat'
    mask_dir = '/home/zf/zz/DATA/mass_roads/train/map'
    test_img_dir = '/home/zf/zz/DATA/mass_roads/test1024/sat'
    test_mask_dir = '/home/zf/zz/DATA/mass_roads/test1024/map'
    record_path = '/home/zf/zz/DATA/mass_roads/train.record'
    test_record_path = '/home/zf/zz/DATA/mass_roads/test.record'
    model_dir = './log/miniunet/'

    num_gpus = 1
    num_classes = 1
    num_steps = 20000
    eval_per_steps = 1000
    score_threshold = 0.5
    # batch size should be larger than 16 if you use batch normalization
    batch_size = 4
    use_batch_norm = False

    hyper_params = {
        'learning_rate_fn': learning_rate_util.cosine_learning_rate(0.1, num_steps, 0.000001),
        'momentum': 0.9,
        'freeze_prefixes': [],
        'weight_decay': 0.0,
    }

    # 1. define loss and metric
    def loss_fn(predictions, labels, params=None):
        pred_logit = predictions['logit']
        labels = tf.reshape(labels, [-1])
        flat_logit = tf.reshape(pred_logit, [-1, num_classes])
        if num_classes > 1:
            onehot_labels = tf.one_hot(labels, num_classes)
        else:
            onehot_labels = tf.reshape(labels, [-1, 1])

        bpn_weights = balance_positive_negative_weight(labels, positive_weight=1.,
                                                       negative_weight=1.)
        # bpn_weights = 1e-3 * bpn_weights
        ce_loss = tf.losses.sigmoid_cross_entropy(onehot_labels, flat_logit, weights=bpn_weights[:, None])
        pred_scores = tf.sigmoid(tf.reshape(flat_logit, [-1]))
        # debug
        # pred_scores = tf.Print(pred_scores, [pred_scores], message='pred_score = ')
        dice_loss_v = dice_loss(pred_scores, labels)
        tf.summary.scalar('loss/cross_entropy_loss', ce_loss)
        tf.summary.scalar('loss/dice_loss', dice_loss_v)
        # add metric for training
        # mean iou
        pred_class = tf.to_float(tf.greater_equal(pred_scores, score_threshold))
        miou = tf.metrics.mean_iou(labels, tf.reshape(pred_class, [-1]), num_classes=2)
        miou_v = compute_mean_iou(None, miou[1])
        tf.identity(miou_v, 'train_miou')
        tf.summary.scalar('train/miou', miou_v)
        # postive iou
        p_iou = positive_iou(labels, tf.reshape(pred_class, [-1]), num_classes=2)
        p_iou_v = compute_positive_iou(None, p_iou[1])
        tf.identity(p_iou_v, 'train_piou')
        tf.summary.scalar('train/piou', p_iou_v)
        # negative iou
        n_iou = negative_iou(labels, tf.reshape(pred_class, [-1]), num_classes=2)
        n_iou_v = compute_negative_iou(None, n_iou[1])
        tf.identity(n_iou_v, 'train_niou')
        tf.summary.scalar('train/niou', n_iou_v)
        # add pr curve
        pr_curve('train/prc', tf.cast(labels, tf.bool), pred_scores, num_thresholds=201)

        return ce_loss + dice_loss_v

    def metric_fn(predictions, labels, params=None):
        images = params['inputs']
        images = denormalize(images)
        pred_prob = predictions['prob']
        flat_pred_prob = tf.reshape(pred_prob, [-1])
        pred_class = tf.to_float(tf.greater_equal(pred_prob, score_threshold))

        # add summary for prediction results.
        pred_images = pred_class * 255
        vis_preds = tf.concat([tf.cast(images, pred_class.dtype), tf.tile(pred_images, multiples=[1, 1, 1, 3])], axis=2)
        vis_preds = tf.cast(vis_preds, tf.uint8)
        tf.summary.image('prediction', vis_preds, max_outputs=10)

        labels = tf.reshape(labels, [-1])
        pred_class = tf.reshape(pred_class, [-1])
        miou = tf.metrics.mean_iou(labels, pred_class, num_classes=2)
        p_iou = positive_iou(labels, pred_class, num_classes=2)
        pr_curve('eval/prc', tf.cast(labels, tf.bool), flat_pred_prob, num_thresholds=201)
        return {
            'eval/miou': miou,
            'eval/piou': p_iou
        }

    def create_model():
        deepunet = Unet4Block(num_classes=num_classes, use_softmax=False, use_batch_norm=use_batch_norm)

        return deepunet

    # 2. prepare data
    train_dataset = seg_data.SegDataset(image_dir=image_dir,
                                        mask_dir=mask_dir,
                                        record_path=record_path,
                                        crop_size_for_train=(512, 512),
                                        rebuild_record=False)

    test_dataset = seg_data.SegDataset(image_dir=test_img_dir,
                                       mask_dir=test_mask_dir,
                                       record_path=test_record_path,
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
        tensors=['learning_rate', 'train_miou', 'train_piou', 'train_niou'],
        every_n_iter=10,
    )
    # 4. go on the fly
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=num_steps, hooks=[log_hook])
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=eval_per_steps, start_delay_secs=0, throttle_secs=30)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
