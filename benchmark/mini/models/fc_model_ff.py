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


def inference(index, activations, last_level=False):
    # Define model
    if isinstance(activations, (list, tuple)):
        for i, acts in enumerate(activations):
            if i == 0:
                x = acts[index]
            else:
                x += acts[index]
    else:
        x = activations

    size = int(FLAGS.weight_size / FLAGS.num_workers)
    if last_level:
        W = tf.Variable(tf.zeros([size, 1]))
    else:
        W = tf.Variable(tf.zeros([size, FLAGS.weight_size]))
    curr_act = tf.matmul(x, W)
    
    if last_level:
        return tf.split(0, FLAGS.num_workers, curr_act)
    else:
        return tf.split(1, FLAGS.num_workers, curr_act)


def loss(logit, labels, index):
    for i, acts in enumerate(logit):
        if i == 0:
            y = acts[index]
        else:
            y += acts[index]
       
    # y = tf.nn.softmax(y)
    # y = tf.reduce_sum(y, reduction_indices=[1])
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(y),
                                                  # reduction_indices=[1]))
    # return cross_entropy
    return (y - labels) * (y - labels)


def get_run_op():
    prev_act = None
    curr_act = []
    labels = []

    # inference
    for level in range(FLAGS.level):
        for i in range(FLAGS.num_workers):
            with tf.device('/gpu:%d' % i):
                if level == 0:
                    d, l = fake_data(FLAGS.batch_size, FLAGS.num_classes)
                    prev_act = d
                    labels.append(l)
                curr_act.append(
                        inference(i, prev_act, level == (FLAGS.level - 1)))
        prev_act = curr_act
        curr_act = []

    # loss
    steps = []
    for i in range(FLAGS.num_workers):
        with tf.device('/gpu:%d' % i):
            steps.append(loss(prev_act, labels[i], i))
    return steps
