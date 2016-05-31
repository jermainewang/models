
from __future__ import absolute_import

import time
from datetime import datetime
import importlib
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import benchmark.variables as variables
import benchmark.scopes as scopes
import benchmark.ops as ops
import benchmark.scopes as scopes
import benchmark.losses as losses

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1024, """Batch size.""")
tf.app.flags.DEFINE_integer('hidden_size', 4096, """Hidden layer size.""")
tf.app.flags.DEFINE_integer('num_batches', 100, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """Number of gpus to run.""")
tf.app.flags.DEFINE_integer('num_layers', 10, """Number of hidden layers""")
tf.app.flags.DEFINE_bool('enable_trace', False, 'Enable trace')

def get_run_op():
  # Create an optimizer that performs gradient descent.
  #opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  slice_size = FLAGS.hidden_size / FLAGS.num_gpus
  print("Slice size: {}".format(slice_size))
  data = []
  label = []
  for i in xrange(FLAGS.num_gpus):
    with tf.device('/gpu:%d' % i):
      data.append(tf.get_variable(
          name = 'data%d' % i,
          shape=[FLAGS.batch_size, slice_size],
          trainable=False))
      '''
      label.append(tf.get_variable(
          name = 'label%d' % i,
          shape = [FLAGS.batch_size, slice_size],
          trainable=False))
      '''
  last = data
  for i in xrange(FLAGS.num_layers):
    tmp = []
    for j in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % j):
        with tf.variable_scope('fc%d' % i):
          w = tf.get_variable(
              name='w%d' % j,
              shape=[slice_size, FLAGS.hidden_size],
              trainable=True)
          # matmult
          y = tf.matmul(last[j], w)
        # split
        tmp.append(tf.split(split_dim=1, num_split=FLAGS.num_gpus, value=y))
    # reduce
    red = []
    for j in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % j):
        red.append(tf.accumulate_n([s[j] for s in tmp]))
    last = red
    if i == FLAGS.num_layers - 1:
      with tf.control_dependencies(last):
        train_op = tf.no_op()
  init_op = tf.initialize_all_variables()
  return init_op, train_op

def time_tensorflow_run(session, target, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
    run_options = None
    run_metadata = None
    if FLAGS.enable_trace and i == num_steps_burn_in - 1:
      run_options = config_pb2.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

    start_time = time.time()
    _ = session.run(target, options=run_options, run_metadata=run_metadata)
    duration = time.time() - start_time

    if FLAGS.enable_trace and i == num_steps_burn_in - 1:
      tl = tf.python.client.timeline.Timeline(run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      with open('tf_trace.ctf', 'w') as f:
        f.write(ctf)

    if i > num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f speed = %.3f images/sec' %
               (datetime.now(), i - num_steps_burn_in, duration, FLAGS.batch_size / duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))

def main(_):
  init_op, train_op = get_run_op()
  # create session
  sess = tf.Session()
  # Run session
  # init variables
  print('Initialize Variables')
  sess.run(init_op)
  print('Initialize Done')
  # run
  merged = tf.merge_all_summaries()
  writer = tf.train.SummaryWriter('/home/minjie/tmp/modelpar', sess.graph)
  time_tensorflow_run(sess, train_op, 'Training')

if __name__ == '__main__':
  tf.app.run()
