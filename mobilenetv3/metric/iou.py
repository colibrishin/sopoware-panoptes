from PIL import Image
import numpy as np
import tensorflow as tf

class IoU(tf.keras.metrics.Metric):
  def __init__(self, n_classes: int, name='IoU', **kwargs):
    super(IoU, self).__init__(name=name, **kwargs)
    self.union = self.add_weight(name='iou_union', initializer='zeros')
    self.intersection = self.add_weight(name='iou_intersection', initializer='zeros')
    self.n_classes = n_classes

  def split_channel(self, y):
    arrays = []
    for i in range(0, self.n_classes):
      arrays.append(y == i)

    return tf.stack(arrays, axis=-1)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = self.split_channel(y_true)
    y_true = tf.cast(y_true, tf.bool)

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = self.split_channel(y_pred)
    y_pred = tf.cast(y_pred, tf.bool)
    
    y_true_count = tf.reduce_sum(tf.cast(y_true, tf.float32)) 
    y_pred_count = tf.reduce_sum(tf.cast(y_pred, tf.float32))
    
    if y_true_count == 0 and y_pred_count == 0:
      self.intersection.assign_add(1)
      self.union.assign_add(1)
     
    else:
      intersection = tf.logical_and(y_true, y_pred)
      union = tf.logical_or(y_true, y_pred)

      intersection = tf.reduce_sum(tf.cast(intersection, tf.float32))
      union = tf.reduce_sum(tf.cast(union, tf.float32))

      self.intersection.assign_add(intersection)
      self.union.assign_add(union)

  def result(self):
    return self.intersection / self.union

class RoadIoU(tf.keras.metrics.Metric):
  def __init__(self, road_class: int, name='RoadIoU', **kwargs):
    super(RoadIoU, self).__init__(name=name, **kwargs)
    self.union = self.add_weight(name='iou_union', initializer='zeros')
    self.intersection = self.add_weight(name='iou_intersection', initializer='zeros')

    self.ROAD_CLASS = road_class

  def split_channel(self, y):
    arrays = []
    arrays.append(y == self.ROAD_CLASS)

    return tf.stack(arrays, axis=-1)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = self.split_channel(y_true)
    y_true = tf.cast(y_true, tf.bool)

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = self.split_channel(y_pred)
    y_pred = tf.cast(y_pred, tf.bool)

    intersection = tf.logical_and(y_true, y_pred)
    union = tf.logical_or(y_true, y_pred)

    intersection = tf.reduce_sum(tf.cast(intersection, tf.float32))
    union = tf.reduce_sum(tf.cast(union, tf.float32))

    self.intersection.assign_add(intersection)
    self.union.assign_add(union)

  def result(self):
    return self.intersection / self.union

class SidewalkIoU(tf.keras.metrics.Metric):

  def __init__(self, sidewalk_class: int, name='SidewalkIoU', **kwargs):
    super(SidewalkIoU, self).__init__(name=name, **kwargs)
    self.union = self.add_weight(name='iou_union', initializer='zeros')
    self.intersection = self.add_weight(name='iou_intersection', initializer='zeros')

    self.SIDEWALK_CLASS = sidewalk_class

  def split_channel(self, y):
    arrays = []
    arrays.append(y == self.SIDEWALK_CLASS)

    return tf.stack(arrays, axis=-1)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = self.split_channel(y_true)
    y_true = tf.cast(y_true, tf.bool)

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = self.split_channel(y_pred)
    y_pred = tf.cast(y_pred, tf.bool)

    intersection = tf.logical_and(y_true, y_pred)
    union = tf.logical_or(y_true, y_pred)

    intersection = tf.reduce_sum(tf.cast(intersection, tf.float32))
    union = tf.reduce_sum(tf.cast(union, tf.float32))

    self.intersection.assign_add(intersection)
    self.union.assign_add(union)
    
  def result(self):
    return self.intersection / self.union