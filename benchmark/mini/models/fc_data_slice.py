import tensorflow as tf
from fc_data_ff import get_run_op as get_ff_run_op
from fc_data_ff import get_var_device
from fc_data_ff import get_all_devices

FLAGS = tf.app.flags.FLAGS

def all_reduce_gradients(tower_grads, devices):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        split_grads = []
        assert len(grad_and_vars) == FLAGS.num_workers
        # Each GPU splits its own grad
        for i, (g, _) in enumerate(grad_and_vars):
            with tf.device(devices[i]):
                split_grads.append(tf.split(0, FLAGS.num_workers, g))
        # Each GPU gatheres slices of grad from other GPUs to do average.
        for i, dev in enumerate(devices):
            with tf.device(dev):
                x = split_grads[i][i]
                for j in range(FLAGS.num_workers):
                    if i == j:
                        continue
                    x += split_grads[j][i]
                grads.append(x / FLAGS.num_workers)
        grad = tf.concat(0, grads)

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
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    for i in range(FLAGS.num_workers):
        tower_grads.append(opt.compute_gradients(losses[i]))
    grads = all_reduce_gradients(tower_grads, get_all_devices())
    apply_gradient_op = opt.apply_gradients(grads)
    train_op = tf.group(apply_gradient_op)
    return train_op
