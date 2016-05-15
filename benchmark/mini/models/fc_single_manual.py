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


def network(inputs, weights, labels):
    targets = []
    bp_sources = []
    g = tf.get_default_graph()

    assert FLAGS.num_workers == 1
    # FF
    x = inputs
    for i in range(FLAGS.level):
        bp_sources.append(x)
        x = tf.matmul(x, weights[i])
    bp_sources.append(x)
    logit = tf.matmul(x, weights[-1])
    loss = (logit - labels) * (logit - labels)

    # return loss

    # BP
    dx = 2 * (logit - labels)
    for i in reversed(range(FLAGS.level + 1)):
        dw = tf.matmul(tf.transpose(bp_sources[i]), dx)
        dx = tf.matmul(dx, tf.transpose(weights[i]))
        new_w = weights[i] - dw
        targets.append(tf.assign(weights[i], new_w))

    # return targets
    with g.control_dependencies(targets):
        final = tf.no_op()
    return final 


def get_run_op():
    with tf.device('/gpu:0'):
        data, labels = fake_data(FLAGS.batch_size, 1)
        weights = initialize_weights()
        gradient_targets = network(data, weights, labels)
    return gradient_targets
