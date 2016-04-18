"""Timing benchmark for AlexNet inference.

To run, use:
  bazel run -c opt --config=cuda \
      third_party/tensorflow/models/image/alexnet:alexnet_benchmark

Across 100 steps on batch size = 128.

Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch

Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch
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
  images = tf.Variable(
      tf.random_normal(
        [batch_size, 224, 224, 3],
        dtype=tf.float32,
        stddev=1e-1), name='images', trainable=False)
  labels = tf.Variable(
      tf.zeros([batch_size, num_classes]), name='labels', trainable=False)
  return images, labels

def inference(images, num_classes, scope):
  with tf.op_scope([images], scope):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.dropout], is_training=True):
      with tf.variable_scope('conv1'):
        conv1 = ops.conv2d(
            images,
            num_filters_out=64,
            kernel_size=[11, 11],
            stride=4,
            stddev=1e-1)
      with tf.variable_scope('pool1'):
        pool1 = ops.max_pool(
            conv1,
            kernel_size=3,
            stride=2,
            padding='VALID')
      with tf.variable_scope('conv2'):
        conv2 = ops.conv2d(
            pool1,
            num_filters_out=192,
            kernel_size=1,
            stride=1,
            stddev=1e-1)
      with tf.variable_scope('pool2'):
        pool2 = ops.max_pool(
            conv2,
            kernel_size=3,
            stride=2,
            padding='VALID')
      with tf.variable_scope('conv3'):
        conv3 = ops.conv2d(
            pool2,
            num_filters_out=384,
            kernel_size=3,
            stride=1,
            stddev=1e-1)
      with tf.variable_scope('conv4'):
        conv4 = ops.conv2d(
            conv3,
            num_filters_out=256,
            kernel_size=3,
            stride=1,
            stddev=1e-1)
      with tf.variable_scope('conv5'):
        conv5 = ops.conv2d(
            conv4,
            num_filters_out=256,
            kernel_size=3,
            stride=1,
            stddev=1e-1)
      with tf.variable_scope('pool3'):
        pool3 = ops.max_pool(
            conv5,
            kernel_size=3,
            stride=2,
            padding='VALID')
      flattened = ops.flatten(pool3, scope='flatten')
      with tf.variable_scope('fc1'):
        fc1 = ops.fc(
            flattened,
            num_units_out=4096)
      with tf.variable_scope('fc2'):
        fc2 = ops.fc(
            fc1,
            num_units_out=4096)
      with tf.variable_scope('fc3'):
        fc3 = ops.fc(
            fc2,
            activation=None,
            num_units_out=num_classes)
  return fc3

def loss(logits, one_hot_labels, batch_size, scope):
  with tf.op_scope([logits, one_hot_labels], scope, 'CrossEntropyLoss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits,
        one_hot_labels,
        name='xentropy')
  return cross_entropy
