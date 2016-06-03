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
tf.app.flags.DEFINE_bool('enable_trace', False, 'Enable trace')

batch_size = None
slice_size = None
feature_size = None

def get_run_op():
  global batch_size
  global slice_size
  global feature_size
  batch_size = FLAGS.batch_size
  slice_size = FLAGS.hidden_size / FLAGS.num_gpus
  feature_size = slice_size * FLAGS.num_gpus
  print("Slice size: {}".format(slice_size))
  data = []
  for i in xrange(FLAGS.num_gpus):
    with tf.device('/gpu:%d' % i):
      data.append(tf.get_variable(
          name = 'data%d' % i,
          shape=[batch_size, slice_size],
          trainable=False))
  # weights
  w = []
  for i in xrange(FLAGS.num_layers):
    w.append([])
    for j in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % j):
        with tf.variable_scope('fc%d' % i):
          w[i].append(tf.get_variable(
              name='w%d' % j,
              shape=[slice_size,feature_size],
              trainable=True))
  # ff
  fwd = []
  last = data
  for i in xrange(FLAGS.num_layers):
    with tf.name_scope('fc_ff%d' % i):
      fwd.append(last)
      tmp = []
      for j in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % j):
          # matmult
          y = tf.matmul(last[j], w[i][j])
          if FLAGS.num_gpus > 1:
            # split
            tmp.append(tf.split(split_dim=1, num_split=FLAGS.num_gpus, value=y))
          else:
            tmp.append(y)
      if FLAGS.num_gpus > 1:
        # reduce
        red = []
        for j in xrange(FLAGS.num_gpus):
          with tf.device('/gpu:%d' % j):
            red.append(tf.accumulate_n([s[j] for s in tmp]))
        last = red
      else:
        last = tmp
  # bp
  targets = []
  for i in reversed(xrange(FLAGS.num_layers)):
    with tf.name_scope('fc_bp%d' % i):
      # convert col -> rep
      tmp = []
      if FLAGS.num_gpus > 1:
        for j in xrange(FLAGS.num_gpus):
          with tf.device('/gpu:%d' % j):
            tmp.append(tf.concat(concat_dim=1, values=last))
      else:
        tmp = last
      last = []
      for j in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % j):
          with tf.name_scope('bp'):
            # matmult: bp
            dy = tf.matmul(tmp[j], w[i][j], transpose_b=True)
            last.append(dy)
          if i == 0:
            dep = [] # no manual scheduling dep since the last bp is not needed
          else:
            dep = [dy] # add manual dep for better scheduling decision
          with tf.control_dependencies(dep), tf.name_scope('grad'):
            # matmult: grad
            dw = tf.matmul(fwd[i][j], tmp[j], transpose_a=True)
          # update
          targets.append(dw)
  with tf.control_dependencies(targets):
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
               (datetime.now(), i - num_steps_burn_in, duration, batch_size / duration))
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
  writer = tf.train.SummaryWriter(log_dir, sess.graph)
  time_tensorflow_run(sess, train_op, 'Training')

if __name__ == '__main__':
  tf.app.run()
