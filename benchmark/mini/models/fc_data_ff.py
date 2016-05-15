import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def fake_data(batch_size, num_classes):
    batch_size = int(batch_size / FLAGS.num_workers)
    data = tf.Variable(
            tf.random_normal(
                [batch_size, FLAGS.weight_size],
                dtype=tf.float32,
                stddev=1e-1), name='data', trainable=False)
    labels = tf.Variable(
                tf.zeros([batch_size, num_classes]), trainable=False)
    return data, labels


def inference(inputs, weights):
    x = inputs
    for i in range(FLAGS.level):
        x = tf.matmul(x, weights[i])
    x = tf.matmul(x, weights[-1])
    return x


def loss(logit, labels):
    return (logit - labels) * (logit - labels)
    # y = tf.nn.softmax(logit)
    # y = tf.reduce_sum(y, reduction_indices=[1])
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(y),
                                   # reduction_indices=[1]))
    # return cross_entropy


def get_var_device():
    if FLAGS.num_workers == 1 or FLAGS.gpu_aggr:
        return '/gpu:0'
    else:
        return '/cpu:0'


def get_all_devices():
    return ['/gpu:%d' % i for i in range(FLAGS.num_workers)]


def get_run_op():
    losses = []
    weights = []
    with tf.device(get_var_device()):
        for i in range(FLAGS.level):
            weights.append(
                    tf.Variable(
                        tf.zeros([FLAGS.weight_size, FLAGS.weight_size])))
        weights.append(
                    tf.Variable(
                        tf.zeros([FLAGS.weight_size, 1])))
        for i in range(FLAGS.num_workers):
            with tf.device('/gpu:%d' % i):
                data, labels = fake_data(FLAGS.batch_size, 1)
                logit = inference(data, weights)
                _loss = loss(logit, labels)
                losses.append(_loss)
    return losses
