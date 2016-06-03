from __future__ import absolute_import

import time
from datetime import datetime
import importlib
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2

import mini.models as models
from mini.distributed_helper import setup_servers, NOT_CONTINUE

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """Number of gpus to run.""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes""")
tf.app.flags.DEFINE_string('model', 'alexnet', """The model to benchmark.""")

flags = tf.app.flags
flags.DEFINE_string('log_dir', None, 'Log dir')
flags.DEFINE_integer('num_workers', 0, 'Number of workers to run')
flags.DEFINE_integer('worker_index', 0, 'The index of this worker')
flags.DEFINE_string('node_name', None, 'ps or worker')
flags.DEFINE_integer('distributed_mode', 0, 'Run distributed mode or not')
flags.DEFINE_string('worker_grpc_url', None, 'Worker GRPC URL')
flags.DEFINE_integer('level', 10, 'Level count for FC/Conv benchmark')
flags.DEFINE_integer('weight_size', 1024, 'Weight size for FC/Conv benchmark')
flags.DEFINE_integer('gpu_aggr', 0, 'Doing gpu aggr on GPU')
flags.DEFINE_integer('img_h', 128, 'Image height')
flags.DEFINE_integer('img_w', 128, 'Image width')
flags.DEFINE_integer('img_d', 3, 'Image depth')
flags.DEFINE_integer('full_trace', 0, 'Full trace')
flags.DEFINE_integer('no_assign', 0, 'No tf.assign')
flags.DEFINE_float('shared_ratio', 0.0009765625, 'Ratio of shared parameters')
flags.DEFINE_integer('phy_blocks', 1, 'Block count for phy')

# some constants
TOWER_NAME = 'MODEL_TOWER'

def _tower_loss(bench, images, labels, num_classes, scope):
  # Build inference Graph
  logits = bench.inference(images, num_classes, scope=scope)

  # Build the portion of the Graph calculating the losses.
  loss = bench.loss(logits, labels, batch_size=FLAGS.batch_size, scope=scope)
  return loss

def _average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def time_tensorflow_run(session, target, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):

    if FLAGS.full_trace == 1 and i == num_steps_burn_in - 1:
        run_options = config_pb2.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = config_pb2.RunMetadata()
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',
                                              session.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
    else:
        run_options = None
        run_metadata = None

    start_time = time.time()
    _ = session.run(target, options=run_options, run_metadata=run_metadata)
    duration = time.time() - start_time

    if FLAGS.full_trace == 1 and i == num_steps_burn_in - 1:
        tl = tf.python.client.timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('full_trace.ctf', 'w') as f:
            f.write(ctf)

    if i >= num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f speed = %.3f images/sec' %
               (datetime.now(), i - num_steps_burn_in, duration, FLAGS.batch_size * FLAGS.num_gpus / duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))


def run_bench(bench_name):
    bench = importlib.import_module(models.__name__ + '.' + bench_name)
    assert bench is not None

    # create session
    if FLAGS.distributed_mode == 1:
        if setup_servers() == NOT_CONTINUE:
            sys.exit()
        sess, train_op = bench.get_run_op()
    else:
        sess = tf.Session()
        # ops
        train_op = bench.get_run_op()
        '''
        g = tf.get_default_graph()
        if not isinstance(train_op, (list, tuple)):
            train_op = [train_op]
        with g.control_dependencies(train_op):
            final = tf.no_op()
         train_op = final
        '''
        init_op = tf.initialize_all_variables()
        # init variables
        sess.run(init_op)

    # run
    info = 'Training %s+%d+%d+%d+%d' % (bench_name, FLAGS.num_workers,
                                        FLAGS.gpu_aggr, FLAGS.weight_size,
                                        FLAGS.batch_size)
    time_tensorflow_run(sess, train_op, info)


def main(_):
    run_bench(FLAGS.model)

if __name__ == '__main__':
  tf.app.run()
