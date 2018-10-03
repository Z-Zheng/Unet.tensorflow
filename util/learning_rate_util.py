import tensorflow as tf
from tensorflow.python.training import learning_rate_decay


def get_warm_up_params(init_value, dst_value, num_steps):
    slope = (dst_value - init_value) * 1.0 / num_steps

    warmup_steps = range(num_steps)
    warmup_rates = [init_value + slope * step for step in warmup_steps]

    return warmup_steps[1:], warmup_rates


def multi_step_learning_rate(step_list, values, warm_up=False, warm_up_init_value=None, warm_up_steps=None):
    # tf.train.piecewise_constant
    if warm_up:
        warm_up_steps, warm_up_values = get_warm_up_params(warm_up_init_value, values[0], warm_up_steps)
        step_list = warm_up_steps + step_list
        values = warm_up_values + values

    def _nested_func(global_step):
        return learning_rate_decay.piecewise_constant(global_step,
                                                      boundaries=step_list,
                                                      values=values)

    return _nested_func


def cosine_learning_rate(learning_rate, decay_steps, alpha=0.0):
    def _nested_func(global_step):
        return learning_rate_decay.cosine_decay(
            learning_rate=learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            alpha=alpha,
        )

    return _nested_func
