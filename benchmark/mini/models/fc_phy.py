import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def fake_data(batch_size, num_classes):
    data = tf.Variable(
            tf.random_normal(
                [batch_size, FLAGS.weight_size],
                dtype=tf.float32,
                stddev=1e-1), name='data', trainable=False)
    labels = tf.Variable(
                tf.zeros([batch_size, num_classes]), trainable=False)
    return data, labels


def initialize_weights():
    weights = []
    for i in range(FLAGS.level):
        weights.append(
                tf.Variable(
                    tf.zeros([FLAGS.weight_size, FLAGS.weight_size])))
    weights.append(
                tf.Variable(
                    tf.zeros([FLAGS.weight_size, 1])))
    return weights


def inference(inputs, weights):
    x = inputs
    for i in range(FLAGS.level):
        x = tf.matmul(x, weights[i])
    x = tf.matmul(x, weights[-1])
    return x


def loss(logit, labels):
    return (logit - labels) * (logit - labels)


def get_all_devices():
    return ['/gpu:%d' % i for i in range(FLAGS.num_workers)]


# Assume that the column dimension is alway intact.
def get_shape(original_shape, ratio):
    return [int(original_shape[0] * ratio), original_shape[1]]


# Assume that the column dimension is alway intact.
def merge_grads(original_grad, partial_new_grad, var):
    original_shape = original_grad.get_shape().as_list()
    partial_new_shape = partial_new_grad.get_shape().as_list()
    slice_origin = [partial_new_shape[0], 0]
    slice_shape = [original_shape[0] - partial_new_shape[0], original_shape[1]]
    new_grad = tf.concat(
                    0,
                    (partial_new_grad, 
                    tf.slice(original_grad, slice_origin, slice_shape)))
                        
    return tf.assign(var, var - new_grad) 
    

def aggregrate_gradients(tower_grads, devices):
    targets = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        sliced_grads = []
        assert (len(grad_and_vars) == FLAGS.num_workers or
                len(grad_and_vars) == FLAGS.phy_blocks)
        # Each GPU splits its own grad
        for i, (g, v) in enumerate(grad_and_vars):
            with tf.device(devices[i]):
                if FLAGS.shared_ratio != 0:
                    slice_shape = get_shape(g.get_shape().as_list(),
                                            FLAGS.shared_ratio)
                    sliced_grads.append(tf.slice(g, [0, 0], slice_shape))
                else:
                    targets.append(tf.assign(v, v - g))
        if FLAGS.shared_ratio != 0:
            # Each GPU gatheres slices of grad from other GPUs to do average.
            for i, dev in enumerate(devices):
                with tf.device(dev):
                    x = sliced_grads[0]
                    for j in range(1, FLAGS.num_workers):
                        x += sliced_grads[j]
                    x = x / FLAGS.num_workers
                    g, v = grad_and_vars[i]
                    targets.append(merge_grads(g, x, v))
    g = tf.get_default_graph()
    with g.control_dependencies(targets):
        target = tf.no_op()
    return target


def get_run_op():
    tower_grads = []
    devices = get_all_devices()
    losses = []
    if FLAGS.num_workers == 1:
        devices = []
        for i in range(FLAGS.phy_blocks):
            devices.append('/gpu:0')
            with tf.device('/gpu:0'):
                data, labels = fake_data(FLAGS.batch_size, 1)
                weights = initialize_weights()
                logit = inference(data, weights)
                _loss = loss(logit, labels)
                opt = tf.train.GradientDescentOptimizer(learning_rate=0.5,
                                                        name=("opt%d" % i))
                tower_grads.append(opt.compute_gradients(_loss, weights))
    else:
        for i in range(FLAGS.num_workers):
            with tf.device('/gpu:%d' % i):
                data, labels = fake_data(FLAGS.batch_size, 1)
                weights = initialize_weights()
                logit = inference(data, weights)
                _loss = loss(logit, labels)
                opt = tf.train.GradientDescentOptimizer(learning_rate=0.5,
                                                        name=("opt%d" % i))
                tower_grads.append(opt.compute_gradients(_loss, weights))
    return aggregrate_gradients(tower_grads, devices)
