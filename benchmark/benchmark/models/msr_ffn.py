"""MSR 5-layer fully connected for speech
https://github.com/Alexey-Kamenev/Benchmarks/blob/master/caffe/ffn.prototxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from benchmark import ops
from benchmark import scopes
from benchmark import losses

def fake_data(batch_size, num_classes):
  data = tf.Variable(
      tf.random_normal(
        [batch_size, 512],
        dtype=tf.float32,
        stddev=1e-1), name='data', trainable=False)
  labels = tf.Variable(
      tf.zeros([batch_size, num_classes]), name='labels', trainable=False)
  return data, labels

def inference(data, num_classes, scope):
  with tf.op_scope([data], scope):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.dropout], is_training=True):
      with tf.variable_scope('fc1'):
        fc1 = ops.fc(
            data,
            num_units_out=2048,
            activation=tf.nn.sigmoid)
      with tf.variable_scope('fc2'):
        fc2 = ops.fc(
            fc1,
            num_units_out=2048,
            activation=tf.nn.sigmoid)
      with tf.variable_scope('fc3'):
        fc3 = ops.fc(
            fc2,
            num_units_out=2048,
            activation=tf.nn.sigmoid)
      with tf.variable_scope('fc4'):
        fc4 = ops.fc(
            fc3,
            num_units_out=2048,
            activation=tf.nn.sigmoid)
      with tf.variable_scope('fc5'):
        fc5 = ops.fc(
            fc4,
            num_units_out=num_classes,
            activation=None)
  return fc5

def loss(logits, one_hot_labels, batch_size, scope):
  with tf.op_scope([logits, one_hot_labels], scope, 'CrossEntropyLoss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits,
        one_hot_labels,
        name='xentropy')
  return cross_entropy
