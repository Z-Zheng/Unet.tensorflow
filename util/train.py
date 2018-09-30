import tensorflow as tf
from tensorflow.contrib import distribute


def build_estimator(model, num_gpus):
    strategy = distribute.MirroredStrategy(num_gpus=num_gpus)
    config = tf.estimator.RunConfig(train_distribute=strategy)

    estimator = tf.keras.estimator.model_to_estimator(model, config=config)

    return estimator
