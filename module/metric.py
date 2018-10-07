import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import distribute as distribute_lib


def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.

    If running in a `DistributionStrategy` context, the variable will be
    "tower local". This means:

    *   The returned object will be a container with separate variables
        per replica/tower of the model.

    *   When writing to the variable, e.g. using `assign_add` in a metric
        update, the update will be applied to the variable local to the
        replica/tower.

    *   To get a metric's result value, we need to sum the variable values
        across the replicas/towers before computing the final answer.
        Furthermore, the final answer should be computed once instead of
        in every replica/tower. Both of these are accomplished by
        running the computation of the final result value inside
        `tf.contrib.distribute.get_tower_context().merge_call(fn)`.
        Inside the `merge_call()`, ops are only added to the graph once
        and access to a tower-local variable in a computation returns
        the sum across all replicas/towers.

    Args:
      shape: Shape of the created variable.
      dtype: Type of the created variable.
      validate_shape: (Optional) Whether shape validation is enabled for
        the created variable.
      name: (Optional) String name of the created variable.

    Returns:
      A (non-trainable) variable initialized to zero, or if inside a
      `DistributionStrategy` scope a tower-local variable container.
    """
    # Note that synchronization "ON_READ" implies trainable=False.
    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        collections=[
            ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
        ],
        validate_shape=validate_shape,
        synchronization=variable_scope.VariableSynchronization.ON_READ,
        aggregation=variable_scope.VariableAggregation.SUM,
        name=name)


def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """Calculate a streaming confusion matrix.

    Calculates a confusion matrix. For estimation over a stream of data,
    the function creates an  `update_op` operation.

    Args:
      labels: A `Tensor` of ground truth labels with shape [batch size] and of
        type `int32` or `int64`. The tensor will be flattened if its rank > 1.
      predictions: A `Tensor` of prediction results for semantic labels, whose
        shape is [batch size] and type `int32` or `int64`. The tensor will be
        flattened if its rank > 1.
      num_classes: The possible number of labels the prediction task can
        have. This value must be provided, since a confusion matrix of
        dimension = [num_classes, num_classes] will be allocated.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).

    Returns:
      total_cm: A `Tensor` representing the confusion matrix.
      update_op: An operation that increments the confusion matrix.
    """
    # Local variable to accumulate the predictions in the confusion matrix.
    total_cm = metric_variable(
        [num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')

    # Cast the type to int64 required by confusion_matrix_ops.
    predictions = math_ops.to_int64(predictions)
    labels = math_ops.to_int64(labels)
    num_classes = math_ops.to_int64(num_classes)

    # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
        predictions = array_ops.reshape(predictions, [-1])

    if labels.get_shape().ndims > 1:
        labels = array_ops.reshape(labels, [-1])

    if (weights is not None) and (weights.get_shape().ndims > 1):
        weights = array_ops.reshape(weights, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = confusion_matrix.confusion_matrix(
        labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
    update_op = state_ops.assign_add(total_cm, current_cm)
    return total_cm, update_op


def compute_positive_iou(total_cm, name):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0), denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    # If the number of valid entries is 0 (no classes) we return 0.
    result = iou[1]

    return result

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

def positive_iou(labels,
                 predictions,
                 num_classes,
                 weights=None,
                 metrics_collections=None,
                 updates_collections=None,
                 name=None):
    if context.executing_eagerly():
        raise RuntimeError('positive_iou is not supported when '
                           'eager execution is enabled.')

    with variable_scope.variable_scope(name, 'positive_iou',
                                       (predictions, labels, weights)):
        # Check if shape is compatible.
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        total_cm, update_op = _streaming_confusion_matrix(labels, predictions,
                                                          num_classes, weights)

        def positive_iou_v_across_towers(_, v):
            positive_iou_v = compute_positive_iou(v, 'positive_iou')
            if metrics_collections:
                tf.add_to_collections(metrics_collections, positive_iou_v)
            return positive_iou_v

        positive_iou_v = distribute_lib.get_tower_context().merge_call(
            positive_iou_v_across_towers, total_cm)

        if updates_collections:
            tf.add_to_collections(updates_collections, update_op)

        return positive_iou_v, update_op
