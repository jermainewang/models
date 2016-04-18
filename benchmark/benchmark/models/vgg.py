""" VGG19 model """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

def inference(inputs, num_classes, scope):
  with tf.op_scope([inputs], scope, 'vgg19'):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout], is_training=True):
      # block 1
      with tf.variable_scope('block1'):
        with tf.variable_scope('conv1'):
          conv1 = ops.conv2d(
              inputs,
              num_filters_out=64,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('conv2'):
          conv2 = ops.conv2d(
              conv1,
              num_filters_out=64,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('pool'):
          pool = ops.max_pool(
              conv2,
              kernel_size=2,
              stride=2,
              padding='VALID')
      # block 2
      with tf.variable_scope('block2'):
        with tf.variable_scope('conv1'):
          conv1 = ops.conv2d(
              pool,
              num_filters_out=128,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('conv2'):
          conv2 = ops.conv2d(
              conv1,
              num_filters_out=128,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('pool'):
          pool = ops.max_pool(
              conv2,
              kernel_size=2,
              stride=2,
              padding='VALID')
      # block 3
      with tf.variable_scope('block3'):
        with tf.variable_scope('conv1'):
          conv1 = ops.conv2d(
              pool,
              num_filters_out=256,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('conv2'):
          conv2 = ops.conv2d(
              conv1,
              num_filters_out=256,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('conv3'):
          conv3 = ops.conv2d(
              conv2,
              num_filters_out=256,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('pool'):
          pool = ops.max_pool(
              conv3,
              kernel_size=2,
              stride=2,
              padding='VALID')
      # block 4
      with tf.variable_scope('block4'):
        with tf.variable_scope('conv1'):
          conv1 = ops.conv2d(
              pool,
              num_filters_out=512,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('conv2'):
          conv2 = ops.conv2d(
              conv1,
              num_filters_out=512,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('conv3'):
          conv3 = ops.conv2d(
              conv2,
              num_filters_out=512,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('pool'):
          pool = ops.max_pool(
              conv3,
              kernel_size=2,
              stride=2,
              padding='VALID')
      # block 5
      with tf.variable_scope('block5'):
        with tf.variable_scope('conv1'):
          conv1 = ops.conv2d(
              pool,
              num_filters_out=512,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('conv2'):
          conv2 = ops.conv2d(
              conv1,
              num_filters_out=512,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('conv3'):
          conv3 = ops.conv2d(
              conv2,
              num_filters_out=512,
              kernel_size=[3, 3],
              stride=1,
              stddev=1e-1)
        with tf.variable_scope('pool'):
          pool = ops.max_pool(
              conv3,
              kernel_size=2,
              stride=2,
              padding='VALID')
      flattened = ops.flatten(pool, scope='flatten')
      # fc
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
