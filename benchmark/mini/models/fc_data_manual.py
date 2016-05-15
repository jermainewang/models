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
    g = tf.get_default_graph()

    bp_sources = [[] for i in range(FLAGS.num_workers + 1)]
    losses = []
    logits = []

    # FF
    for j in range(FLAGS.num_workers):
        with tf.device('/gpu:%d' % j):
            x = inputs[j]
            for i in range(FLAGS.level):
                bp_sources[j].append(x)
                x = tf.matmul(x, weights[j][i])
            bp_sources[j].append(x)
            logit = tf.matmul(x, weights[j][-1])
            loss = (logit - labels[j]) * (logit - labels[j])
            losses.append(loss)
            logits.append(logit)

    # return losses
    # BP
    dxs = []
    temp_dxs = []
    for j in range(FLAGS.num_workers):
        with tf.device('/gpu:%d' % j):
            dx = 2 * (logits[j] - labels[j])
            dxs.append(dx)

    for i in reversed(range(FLAGS.level + 1)):
        dws = []
        for j in range(FLAGS.num_workers):
            with tf.device('/gpu:%d' % j):
                dw = tf.matmul(tf.transpose(bp_sources[j][i]), dxs[j])
                if i == FLAGS.level:
                    dws.append(dw)
                else:
                    dws.append(tf.split(1, FLAGS.num_workers, dw))
                dx = tf.matmul(dxs[j], tf.transpose(weights[j][i]))
                temp_dxs.append(dx)
                
        dxs = temp_dxs
        temp_dxs = []

        # Update weights
        # Reduce
        if i == FLAGS.level:
            for j in range(FLAGS.num_workers):
                with tf.device('/gpu:%d' % j):
                    dw = sum(dws) / FLAGS.num_workers
                    targets.append(tf.assign(weights[j][i], weights[j][i] - dw))
        else:
            partial_dws = []
            for j in range(FLAGS.num_workers):
                with tf.device('/gpu:%d' % j):
                    partial_dw = dws[0][j]
                    for dw in dws[1:]:
                        partial_dw += dw[j]
                    partial_dw /= FLAGS.num_workers
                    partial_dws.append(partial_dw)

            # Aggregrate
            for j in range(FLAGS.num_workers):
                with tf.device('/gpu:%d' % j):
                    dw = tf.concat(1, partial_dws)
                    targets.append(tf.assign(weights[j][i], weights[j][i] - dw))

    # return targets
    with g.control_dependencies(targets):
        final = tf.no_op()
    return final 


def get_run_op():
    assert FLAGS.num_workers == 2
    datum = []
    labels = []
    weights = []
    for i in range(FLAGS.num_workers):
        with tf.device('/gpu:%d' % i):
            data, label = fake_data(FLAGS.batch_size, 1)
            datum.append(data)
            labels.append(label)
            weights.append(initialize_weights())
    gradient_targets = network(datum, weights, labels)
    return gradient_targets
