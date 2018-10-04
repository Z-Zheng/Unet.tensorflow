import tensorflow as tf
from module.unet import Unet
from util import estimator_util
from data import seg_data
from util import learning_rate_util


def main():
    # todo: use a config system to manage
    # config
    image_dir = '/media/zhengzhuo/0BE616A90BE616A9/Unet.tensorflow/test_images/images'
    mask_dir = '/media/zhengzhuo/0BE616A90BE616A9/Unet.tensorflow/test_images/masks'
    record_path = '/media/zhengzhuo/0BE616A90BE616A9/Unet.tensorflow/test_images/train.record'
    model_dir = './tmp'

    num_gpus = 1
    num_classes = 2
    num_steps = 20000
    batch_size = 4

    hyper_params = {
        'learning_rate_fn': learning_rate_util.cosine_learning_rate(0.1, num_steps, 0.000001),
        'momentum': 0.9,
        'freeze_prefixes': []
    }
    # 1. define loss and metric
    def loss_fn(predictions, labels):
        pred_logit = predictions['logit']
        labels = tf.reshape(labels, [-1])
        flat_logit = tf.reshape(pred_logit, [-1, num_classes])
        onehot_labels = tf.one_hot(labels, num_classes)
        ce_loss = tf.losses.softmax_cross_entropy(onehot_labels, flat_logit)
        tf.summary.scalar('loss/cross_entropy_loss', ce_loss)
        return tf.losses.softmax_cross_entropy(onehot_labels, flat_logit)

    def metric_fn(predictions, labels):
        pred_prob = predictions['prob']
        pred_class = tf.argmax(pred_prob, axis=-1)
        labels = tf.reshape(labels, [-1])
        pred_class = tf.reshape(pred_class, [-1])
        miou = tf.metrics.mean_iou(labels, pred_class, num_classes=num_classes)
        tf.summary.scalar('eval/miou', miou)
        return {
            'miou': miou
        }
    # 2. prepare data
    train_dataset = seg_data.SegDataset(image_dir=image_dir,
                                        mask_dir=mask_dir,
                                        record_path=record_path,
                                        crop_size_for_train=(512, 512),
                                        rebuild_record=False)

    test_dataset = seg_data.SegDataset(image_dir=image_dir,
                                       mask_dir=mask_dir,
                                       record_path=record_path,
                                       crop_size_for_train=None,
                                       rebuild_record=False)

    train_input_fn = train_dataset.get_input_fn(batch_size=batch_size, epochs=-1, training=True)
    eval_input_fn = test_dataset.get_input_fn(batch_size=1, epochs=1, training=False)

    # 3. prepare model
    model = Unet(num_classes=num_classes, use_softmax=True)

    estimator = estimator_util.build_estimator(model,
                                               model_dir,
                                               num_gpus=num_gpus,
                                               hpyerparams=hyper_params,
                                               loss_fn=loss_fn,
                                               metric_fn=metric_fn)

    log_hook = tf.train.LoggingTensorHook(
        tensors=['learning_rate'],
        every_n_iter=10,
    )

    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=num_steps, hooks=[log_hook])
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=2000)

    # 4. go on the fly
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
