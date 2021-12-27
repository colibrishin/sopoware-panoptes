import tensorflow as tf
import numpy as np

class MeanIoUArgMax(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)

class MeanIoUArgMaxWithClass(MeanIoUArgMax):
  '''
  From Keras MeanIoU, Argmax the y_pred
  pick one class IoU
  '''
  def __init__(self, num_classes, n_class, name='MeanIoUArgMaxWithClass', dtype=None):
    super(MeanIoUArgMaxWithClass, self).__init__(name=name, dtype=dtype)
    self.n_class = n_class
    self.num_classes = num_classes

    # Variable to accumulate the predictions in the confusion matrix.
    self.total_cm = self.add_weight(
        'total_confusion_matrix',
        shape=(num_classes, num_classes),
        initializer=tf.compat.v1.zeros_initializer)

  def result(self):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.cast(
        tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)[self.n_class]
    sum_over_col = tf.cast(
        tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)[self.n_class]
    true_positives = tf.cast(
        tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype)[self.n_class]

    # sum_over_row + sum_over_col =
    #     2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(
        tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))

    iou = tf.math.divide_no_nan(true_positives, denominator)

    return tf.math.divide_no_nan(
        tf.reduce_sum(iou, name='mean_iou'), num_valid_entries)

  def get_config(self):
    config = {'num_classes': self.num_classes, 'class': self.n_class}
    base_config = super(MeanIoUArgMaxWithClass, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))