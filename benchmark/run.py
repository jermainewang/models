from __future__ import absolute_import

import time
from datetime import datetime
import importlib
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import benchmark.models as models
import benchmark.variables as variables
import benchmark.scopes as scopes

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """Number of gpus to run.""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes""")
tf.app.flags.DEFINE_string('model', 'alexnet', """The model to benchmark.""")

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
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i > num_steps_burn_in:
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

def get_run_op(bench):
  var_device = '/cpu:0' if FLAGS.num_gpus > 1 else '/gpu:0'
  with tf.device(var_device):
    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    # calculate the gradients for each model tower
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
          images_batch, labels_batch = bench.fake_data(FLAGS.batch_size, FLAGS.num_classes)
          # need to force all variables on CPU if with multiple GPUs
          with scopes.arg_scope([variables.variable], device=var_device):
            loss = _tower_loss(bench, images_batch, labels_batch, FLAGS.num_classes, scope)
          # Reuse variables for next tower
          tf.get_variable_scope().reuse_variables()
          # calculate gradients
          grads = opt.compute_gradients(loss)
          # keep it
          tower_grads.append(grads)
    # average gradients
    grads = _average_gradients(tower_grads)
    # apply (update)
    apply_gradient_op = opt.apply_gradients(grads)
    train_op = tf.group(apply_gradient_op)
    # Build initialization operation
    init_op = tf.initialize_all_variables()

    return init_op, train_op

def run_bench(bench_name):
  bench = importlib.import_module(models.__name__ + '.' + bench_name)
  assert bench is not None
  init_op, train_op = get_run_op(bench)

  # create session
  sess = tf.Session(config=tf.ConfigProto())

  # init variables
  print('Initialize Variables')
  sess.run(init_op)
  print('Initialize Done')

  # run
  time_tensorflow_run(sess, train_op, 'Training')

def main(_):
  run_bench(FLAGS.model)

if __name__ == '__main__':
  tf.app.run()
