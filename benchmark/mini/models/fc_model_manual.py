import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def fake_data(batch_size, num_classes):
    data = tf.Variable(
            tf.random_normal(
                [batch_size, int(FLAGS.weight_size / FLAGS.num_workers)],
                dtype=tf.float32,
                stddev=1e-1), name='data', trainable=False)

    labels = tf.Variable(
                tf.zeros([int(batch_size / FLAGS.num_workers), num_classes]),
                         trainable=False)
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


def network(inputs, labels):
    '''
    FF
    '''
    size = int(FLAGS.weight_size / FLAGS.num_workers)
    activations = inputs
    temp_activations = []
    weights = [[] for i in range(FLAGS.level + 1)]
    bp_sources = [[] for i in range(FLAGS.level + 1)]

    # FCs
    for i in range(FLAGS.level):
        for j in range(FLAGS.num_workers):
            with tf.device('/gpu:%d' % j):
                if i == 0:
                    x = activations[j]
                else:
                    x = activations[0][j]
                    for acts in activations[1:]:
                        x += acts[j]
                W = tf.Variable(tf.zeros([size, FLAGS.weight_size]))
                weights[i].append(W)
                bp_sources[i].append(x)
                curr_act = tf.matmul(x, W)
                temp_activations.append(tf.split(1, FLAGS.num_workers, curr_act))
        activations = temp_activations
        temp_activations = []
    # The last FC
    for j in range(FLAGS.num_workers):
        with tf.device('/gpu:%d' % j):
            x = activations[0][j]
            for acts in activations[1:]:
                x += acts[j]
            W = tf.Variable(tf.zeros([size, 1]))
            weights[-1].append(W)
            bp_sources[-1].append(x)
            curr_act = tf.matmul(x, W)
            temp_activations.append(tf.split(0, FLAGS.num_workers, curr_act))
    activations = temp_activations
    temp_activations = []

    # Loss
    logits = []
    losses = []
    for j in range(FLAGS.num_workers):
        with tf.device('/gpu:%d' % j):
            logits.append(activations[0][j])
            for acts in activations[1:]:
                logits[-1] += acts[j]
        loss = (logits[-1] - labels[j]) * (logits[-1] - labels[j])
        losses.append(loss)
    # return losses

    '''
    BP
    '''
    dxs = []
    temp_dxs = []
    for j in range(FLAGS.num_workers):
        with tf.device('/gpu:%d' % j):
            dx = 2 * (logits[j] - labels[j])
            dxs.append(dx)
    for j in range(FLAGS.num_workers):
        with tf.device('/gpu:%d' % j):
            dx = tf.concat(0, dxs)
            temp_dxs.append(dx)
    dxs = temp_dxs
    temp_dxs = []

    targets = []
    g = tf.get_default_graph()
    for i in reversed(range(FLAGS.level + 1)):
        for j in range(FLAGS.num_workers):
            with tf.device('/gpu:%d' % j):
                dw = tf.matmul(tf.transpose(bp_sources[i][j]), dxs[j])
                dx = tf.matmul(dxs[j], tf.transpose(weights[i][j]))
                dxs[j] = dx
                new_w = weights[i][j] - dw
                targets.append(tf.assign(weights[i][j], new_w))
        for j in range(FLAGS.num_workers):
            with tf.device('/gpu:%d' % j):
                dx = tf.concat(1, dxs)
                temp_dxs.append(dx)
        dxs = temp_dxs
        temp_dxs = []

    # return targets
    with g.control_dependencies(targets):
        final = tf.no_op()
    return final 


def get_run_op():
    assert FLAGS.num_workers == 2
    datum = []
    labels = []
    for i in range(FLAGS.num_workers):
        with tf.device('/gpu:%d' % i):
            data, label = fake_data(FLAGS.batch_size, 1)
            datum.append(data)
            labels.append(label)
    gradient_targets = network(datum, labels)
    return gradient_targets
