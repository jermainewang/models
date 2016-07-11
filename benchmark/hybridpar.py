from __future__ import absolute_import

import time
from datetime import datetime
import importlib
import math
import os
from six.moves import range  # pylint: disable=redefined-builtin
import numpy as np

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
tf.app.flags.DEFINE_integer('num_layers', 10, """Number of hidden layers""")
tf.app.flags.DEFINE_integer('num_mp', 1, """Number of model parallelism.""")
tf.app.flags.DEFINE_integer('num_dp', 1, """Number of data parallelism.""")
tf.app.flags.DEFINE_bool('enable_trace', False, 'Enable trace')

def wname(layerid, splitid):
  return 'w%d%d' % (layerid, splitid)

class Model:
  def __init__(self, num_splits):
    # init weights
    self.weights = {}
    self.num_splits = num_splits
    self.slice_size = FLAGS.hidden_size / num_splits
    print("Slice size: {}".format(self.slice_size))
    self.feature_size = slice_size * num_splits
    for i in range(FLAGS.num_layers):
      for j in range(num_splits):
        with tf.device('/cpu'):
          name = wname(i, j)
          self.weights[name] = tf.get_variable(name=name,
                                               shape=[self.slice_size, self.feature_size],
                                               trainable=True))

  def fake_data(self, batch_size, gpus):
    data = []
    for gid in self.gpus:
      with tf.device('/gpu:%d' % gid):
        data.append(tf.get_variable(name = 'data%d' % gid,
                                    shape=[batch_size, self.slice_size],
                                    trainable=False))
    return data

  def run(self, data, gpus):
    # grads
    grads = {}
    # ff
    fwd = []
    last = data
    for i in range(FLAGS.num_layers):
      with tf.name_scope('fc_ff%d' % i):
        fwd.append(last)
        y_split = []
        for j, gid in enumerate(gpus):
          with tf.device('/gpu:%d' % gid):
            # matmult
            y = tf.matmul(last[j], self.weights[wname(i, j)])
            # split
            y_split.append(tf.split(split_dim=1, num_split=len(gpus), value=y))
      with tf.name_scope('red2col%d' % i):
        # reduce
        red = []
        for j, gid in enumerate(gpus):
          with tf.device('/gpu:%d' % gid):
            red.append(tf.accumulate_n([s[j] for s in y_split]))
        last = red
    # bp
    for i in reversed(range(FLAGS.num_layers)):
      with tf.name_scope('col2rep%d' % i):
        # convert col -> rep
        dy_rep = []
        for j, gid in enumerate(gpus):
          with tf.device('/gpu:%d' % gid):
            dy_rep.append(tf.concat(concat_dim=1, values=last))
      last = []
      with tf.name_scope('fc_bp%d' % i):
        for j, gid in enumerate(gpus):
          with tf.device('/gpu:%d' % gid):
            # matmult: bp
            dy = tf.matmul(dy_rep[j], self.weights[wname(i, j)], transpose_b=True)
            last.append(dy)
      with tf.name_scope('fc_grad%d' % i):
        for j, gid in enumerate(gpus):
          with tf.device('/gpu:%d' % gid):
            # matmult: grad
            grads[wname(i, j)] = tf.matmul(fwd[i][j], dy_rep[j], transpose_a=True)

def train():
  gpus = np.arange(FLAGS.num_dp * FLAGS.num_mp).reshape([FLAGS.num_dp, FLAGS.num_mp]).tolist()
  model = Model(FLAGS.num_mp)
  all_grads = {}
  for i in range(FLAGS.num_dp):
    model.reset()
    data = model.fake_data(FLAGS.batch_size / FLAGS.num_dp, gpus[i])
    grads = model.run(data, gpus[i])
    for k, v in grads.items():
      if not k in all_grads:
        all_grads[k] = []
      all_grads[k].append(v)
  # accumulate grads
  targets = []
  with tf.device('/cpu'), tf.name_scope('grad_accum'):
    for k, v in all_grads.items():
      targets.append(tf.accumulate_n(v))
  # ops
  with tf.control_dependencies(targets):
    train_op = tf.no_op()
  init_op = tf.initialize_all_variables()
  return init_op, train_op

def time_tensorflow_run(session, target, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in range(FLAGS.num_batches + num_steps_burn_in):
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
  init_op, train_op = train()
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
