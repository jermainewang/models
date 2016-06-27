from __future__ import absolute_import

import time
from datetime import datetime
import importlib
import math
import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import benchmark.variables as variables
import benchmark.scopes as scopes
import benchmark.ops as ops
import benchmark.scopes as scopes
import benchmark.losses as losses

log_dir = os.path.join(os.environ['HOME'], 'tmp', 'modelpar')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 1024, """Batch size.""")
tf.app.flags.DEFINE_integer('hidden_size', 4096, """Hidden layer size.""")
tf.app.flags.DEFINE_integer('num_batches', 100, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """Number of gpus to run.""")
tf.app.flags.DEFINE_integer('num_layers', 10, """Number of hidden layers""")
tf.app.flags.DEFINE_integer('num_cuts', 4, """Number of cuts on batch size to allow pipelining""")
tf.app.flags.DEFINE_bool('enable_trace', False, 'Enable trace')

def make_data(batch_size, slice_size):
  data = []
  for i in xrange(FLAGS.num_gpus):
    with tf.device('/gpu:%d' % i):
      data.append(tf.zeros(shape=[batch_size, slice_size]))
  return data

def make_weights(weight_shape):
  # weights
  w = []
  grads = []
  for i in xrange(FLAGS.num_layers):
    w.append([])
    grads.append([])
    for j in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % j):
        with tf.variable_scope('fc%d' % i):
          w[i].append(tf.get_variable(
              name='w%d' % j,
              shape=weight_shape,
              trainable=True))
          grads[i].append(tf.zeros(shape=weight_shape))
  return w, grads

def ff_bp(data, w, grads, ff_deps, bp_deps):
  new_ff_deps = []
  new_bp_deps = []
  # ff
  fwd = []
  last = data
  for i in xrange(FLAGS.num_layers):
    with tf.name_scope('fc_ff%d' % i):
      fwd.append(last)
      tmp = []
      new_ff_deps.append([])
      for j in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % j), tf.control_dependencies([ff_deps[i][j]]):
          # matmult
          y = tf.matmul(last[j], w[i][j])
          # split
          y_split = tf.split(split_dim=1, num_split=FLAGS.num_gpus, value=y)
          tmp.append(y_split)
          new_ff_deps[i].append(y)
      # reduce
      red = []
      for j in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % j):
          red.append(tf.accumulate_n([s[j] for s in tmp]))
      last = red
  # bp
  for i in reversed(xrange(FLAGS.num_layers)):
    with tf.name_scope('fc_bp%d' % i):
      # convert col -> rep
      tmp = []
      for j in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % j):
          tmp.append(tf.concat(concat_dim=1, values=last))
      last = []
      for j in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % j):
          with tf.name_scope('bp'):
            # matmult: bp
            dy = tf.matmul(tmp[j], w[i][j], transpose_b=True)
            last.append(dy)
            # matmult: grad
            dw = tf.matmul(fwd[i][j], tmp[j], transpose_a=True)
          # update
          grads[i][j] += dw
  return new_ff_deps, new_bp_deps

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
  assert(FLAGS.num_gpus > 1)
  slice_size = int(FLAGS.hidden_size / FLAGS.num_gpus)
  feature_size = slice_size * FLAGS.num_gpus
  print("Slice size: {} Feature size: {}".format(slice_size, feature_size))
  weight_shape = [slice_size, feature_size]

  # create graph
  weights, grads = make_weights(weight_shape)
  ff_deps = [[tf.no_op() for j in range(FLAGS.num_gpus)] for i in range(FLAGS.num_layers)]
  bp_deps = [[tf.no_op() for j in range(FLAGS.num_gpus)] for i in range(FLAGS.num_layers)]
  for i in range(FLAGS.num_cuts):
    with tf.name_scope('data_cut%d' % i):
      data = make_data(FLAGS.batch_size / FLAGS.num_cuts, slice_size)
    with tf.name_scope('model_cut%d' % i):
      ff_deps, bp_deps = ff_bp(data, weights, grads, ff_deps, bp_deps)

  # create session
  sess = tf.Session()
  # init variables
  print('Initialize Variables')
  sess.run(tf.initialize_all_variables())
  print('Initialize Done')
  # run
  merged = tf.merge_all_summaries()
  writer = tf.train.SummaryWriter(log_dir, sess.graph)
  grads_flatten = sum(grads, [])
  with tf.control_dependencies(grads_flatten):
    train_op = tf.no_op()
  time_tensorflow_run(sess, train_op, 'Training')

if __name__ == '__main__':
  tf.app.run()
