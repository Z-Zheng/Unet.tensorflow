import tensorflow as tf
from tensorflow.contrib import distribute
from module import loss
from util import learning_rate_util
from functools import partial

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_estimator(create_model_func,
                    model_dir,
                    num_gpus,
                    hpyerparams,
                    loss_fn,
                    metric_fn=None
                    ):
    # issue #22550
    strategy = distribute.MirroredStrategy(num_gpus=num_gpus)
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    config = tf.estimator.RunConfig(train_distribute=strategy,
                                    log_step_count_steps=10,
                                    keep_checkpoint_max=20,
                                    session_config=session_config)

    model_fn = build_model_fn(create_model_func, hpyerparams, loss_fn=loss_fn, metric_fn=metric_fn)
    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=model_dir,
        config=config
    )
    return estimator


def build_model_fn(create_model_fn, hpyerparams, loss_fn, metric_fn=None):
    def _model_fn(features, labels, mode, params=None):
        model = create_model_fn()
        prediction = model(features)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=prediction,
            )
        main_losses = loss_fn(predictions=prediction, labels=labels)
        metric_ops = metric_fn(predictions=prediction, labels=labels, params={
            'inputs': features
        })
        if mode == tf.estimator.ModeKeys.TRAIN:
            # get hyper-parameters
            hps = HyperParams(hpyerparams)
            learning_rate_fn = hps.get('learning_rate_fn')
            momentum = hps.get('momentum')
            freeze_prefixes = hps.get('freeze_prefixes')
            weight_decay = hps.get('weight_decay')
            # create global_step
            global_step = tf.train.get_or_create_global_step()
            lr = learning_rate_fn(global_step)
            tf.identity(lr, 'learning_rate')
            tf.summary.scalar('learning_rate', lr)
            # create opt
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
            # freeze vars
            trainable_var_list = model.trainable_variables
            trainable_var_list = filter_var_list(trainable_var_list, freeze_prefixes)
            if len(trainable_var_list) == 0:
                logger.warning('There is no variable to be trained.')
            # add l2 weight decay loss
            l2_loss = loss.l2_weight_decay(trainable_var_list, weight_decay, loss_filter_fn=None)
            tf.summary.scalar('loss/l2_regularization_loss', l2_loss)
            total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('loss/total_loss', total_loss)

            grads_and_vars = optimizer.compute_gradients(total_loss, var_list=trainable_var_list)
            # todo: support grad operations
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            return tf.estimator.EstimatorSpec(
                mode,
                loss=tf.losses.get_total_loss(),
                train_op=train_op
            )
        if mode == tf.estimator.ModeKeys.EVAL:

            if metric_fn is None:
                raise ValueError('The mode is EVAL, but the metric_fn is None.')

            return tf.estimator.EstimatorSpec(
                mode,
                loss=tf.losses.get_total_loss(),
                eval_metric_ops=metric_ops
            )

    return _model_fn


def filter_var_list(var_list, prefixes):
    ret_list = []
    for var in var_list:
        keep = True
        for prefix in prefixes:
            if var.op.name.startswith(prefix):
                keep = False
                continue

        if keep:
            ret_list.append(var)
    return ret_list


class HyperParams(object):
    def __init__(self, hyper_param_dict=None):
        if hyper_param_dict is None:
            self.data = {}
        else:
            self.data = hyper_param_dict

        self.defalut_dict = {
            'learning_rate': learning_rate_util.cosine_learning_rate(0.1, 20000, 0.000001),
            'momentum': 0.9,
            'freeze_prefixes': [],
            'weight_decay': 5e-4
        }

    def get(self, key):
        if key not in self.data:
            return self.defalut_dict[key]
        else:
            return self.data[key]


def get_hyperparam(key, hpyerparams, default_value):
    if key not in hpyerparams:
        return default_value
    else:
        return hpyerparams[key]


def export_model():
    pass
