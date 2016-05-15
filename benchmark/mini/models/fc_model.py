import tensorflow as tf
from fc_model_ff import get_run_op as get_ff_run_op

FLAGS = tf.app.flags.FLAGS

def get_run_op():
    steps = get_ff_run_op()
    return [tf.train.GradientDescentOptimizer(0.5).minimize(st) for st in steps]
