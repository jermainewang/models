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

def _res_block(inputs, num_filters_start, half_input):
  print (inputs.get_shape())
  num_filters_in = inputs.get_shape()[-1]
  num_filters_out = num_filters_start * 4
  need_branch1 = (num_filters_in != num_filters_out)
  # TODO tf does not support conv2d with stride > kernel_size
  # Workaround: use a pooling to replace this downsampling
  #stride1 = 2 if half_input else 1
  stride1 = 1
  if half_input:
    with tf.variable_scope('downsample'):
      inputs = ops.max_pool(
          inputs,
          kernel_size=3,
          stride=2,
          padding='SAME')
  if need_branch1:
    with tf.variable_scope('branch1'):
      branch1 = ops.conv2d(
          inputs,
          num_filters_out=num_filters_out,
          kernel_size=1,
          stride=stride1)
  with tf.variable_scope('branch2'):
    with tf.variable_scope('a'):
      branch2a = ops.conv2d(
          inputs,
          num_filters_out=num_filters_start,
          kernel_size=1,
          stride=stride1)
    with tf.variable_scope('b'):
      branch2b = ops.conv2d(
          branch2a,
          num_filters_out=num_filters_start,
          kernel_size=3,
          stride=1)
    with tf.variable_scope('c'):
      branch2c = ops.conv2d(
          branch2b,
          num_filters_out=num_filters_out,
          kernel_size=1,
          stride=1,
          activation=None)
  branch_sum = branch2c + branch1 if need_branch1 else branch2c
  ret = tf.nn.relu(branch_sum)
  return ret

def _res_group(inputs, num_filters_start, num_blocks, first_group=False):
  for i in range(num_blocks):
    with tf.variable_scope('blk%d' % i):
      half_input = (i == 0 and not first_group)
      inputs = _res_block(inputs, num_filters_start=num_filters_start, half_input=half_input)
  return inputs

def inference(inputs, num_classes, scope):
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  with tf.op_scope([inputs], scope, 'resnet50'):
    # TODO how to implement scale layer?
    with scopes.arg_scope([ops.conv2d], batch_norm_params=batch_norm_params):
      with tf.variable_scope('conv1'):
        conv1 = ops.conv2d(
            inputs,
            num_filters_out=64,
            kernel_size=7,
            stride=2)
      with tf.variable_scope('poo1'):
        pool1 = ops.max_pool(
            conv1,
            kernel_size=3,
            stride=2,
            padding='SAME')
      with tf.variable_scope('res2'):
        res2 = _res_group(pool1, num_filters_start=64, num_blocks=3, first_group=True)
      with tf.variable_scope('res3'):
        res3 = _res_group(res2, num_filters_start=128, num_blocks=4)
      with tf.variable_scope('res4'):
        res4 = _res_group(res3, num_filters_start=256, num_blocks=6)
      with tf.variable_scope('res5'):
        res5 = _res_group(res4, num_filters_start=512, num_blocks=3)
      with tf.variable_scope('pool5'):
        pool5 = ops.avg_pool(
            res5, 
            kernel_size=7,
            stride=1,
            padding='SAME')
      flattened = ops.flatten(pool5, scope='flatten')
      with tf.variable_scope('fc'):
        fc = ops.fc(
            flattened,
            activation=None,
            num_units_out=num_classes)
  return fc

def loss(logits, one_hot_labels, batch_size, scope):
  with tf.op_scope([logits, one_hot_labels], scope, 'CrossEntropyLoss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits,
        one_hot_labels,
        name='xentropy')
  return cross_entropy
