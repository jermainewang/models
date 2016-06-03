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

def get_run_op():
  slice_size = FLAGS.batch_size / FLAGS.num_gpus
  print("Slice size: {}".format(slice_size))
  data = []
  for i in xrange(FLAGS.num_gpus):
    with tf.device('/gpu:%d' % i):
      data.append(tf.get_variable(
          name = 'data%d' % i,
          shape=[slice_size, FLAGS.hidden_size],
          trainable=False))
  # weights
  w = []
  for i in xrange(FLAGS.num_layers):
    with tf.device('/gpu:%d' % (i % FLAGS.num_gpus)):
      with tf.variable_scope('fc%d' % i):
        w.append(tf.get_variable(
            name='w',
            shape=[FLAGS.hidden_size, FLAGS.hidden_size],
            trainable=True))
  # ff
  def fwd_bwd(which):
    fwd = []
    last = data[which]
    for i in xrange(FLAGS.num_layers):
      with tf.name_scope('fc_ff%d' % i):
        fwd.append(last)
        # matmult
        last = tf.matmul(last, w[i])
    # bp
    targets = []
    for i in reversed(xrange(FLAGS.num_layers)):
      with tf.name_scope('fc_bp%d' % i):
        with tf.name_scope('grad'):
          # matmult: grad
          dw = tf.matmul(fwd[i], last, transpose_a=True)
        with tf.name_scope('bp'):
          # matmult: bp
          last = tf.matmul(last, w[i], transpose_b=True)
        # update
        targets.append(dw)
    return targets

  tower_grads = []
  for i in xrange(FLAGS.num_gpus):
    with tf.device('/gpu:%d' % i), tf.name_scope('tower%d' % i):
      tower_grads.append(list(reversed(fwd_bwd(i))))
  targets = []
  if FLAGS.num_gpus > 1:
    # accumulation
    for i in xrange(FLAGS.num_layers):
      with tf.device('/gpu:%d' % (i % FLAGS.num_gpus)):
        with tf.name_scope('grad_accum%d' % i):
          targets.append(tf.accumulate_n([t[i] for t in tower_grads]))
  else:
    targets = tower_grads[0]
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
               (datetime.now(), i - num_steps_burn_in, duration, FLAGS.batch_size / FLAGS.num_gpus * FLAGS.num_gpus / duration))
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
