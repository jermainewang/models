import tensorflow as tf
from fc_data_ff import get_run_op as get_ff_run_op
from fc_data_ff import get_var_device

FLAGS = tf.app.flags.FLAGS

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


def get_run_op():
    losses = get_ff_run_op()
    tower_grads = []
    with tf.device(get_var_device()):
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    for i in range(FLAGS.num_workers):
        with tf.device('/gpu:%d' % i):
            tower_grads.append(opt.compute_gradients(losses[i]))
    with tf.device(get_var_device()):
        grads = _average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads)
        train_op = tf.group(apply_gradient_op)
    return train_op
