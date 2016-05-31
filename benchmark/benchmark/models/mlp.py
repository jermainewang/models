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

FLAGS = tf.app.flags.FLAGS

def fake_data(batch_size, num_classes):
  data = tf.Variable(
      tf.random_normal(
        [batch_size, FLAGS.layer_size],
        dtype=tf.float32,
        stddev=1e-1), name='data', trainable=False)
  labels = tf.Variable(
      tf.zeros([batch_size, num_classes]), name='labels', trainable=False)
  return data, labels

def inference(data, num_classes, scope):
  with tf.op_scope([data], scope):
    last = data
    for i in range(FLAGS.num_layers - 1):
      with tf.variable_scope('fc%d' % i):
        w = tf.get_variable(
            name="w",
            shape=[FLAGS.layer_size, FLAGS.layer_size],
            trainable=True)
        last = tf.matmul(last, w)
    with tf.variable_scope('fc%d' % (FLAGS.num_layers - 1)):
      w = tf.get_variable(
          name="w",
          shape=[FLAGS.layer_size, num_classes],
          trainable=True)
      last = tf.matmul(last, w)
  return last

def loss(logits, one_hot_labels, batch_size, scope):
  with tf.op_scope([logits, one_hot_labels], scope, 'L2Loss'):
    ret = tf.reduce_sum((logits - one_hot_labels) ** 2)
  return ret
