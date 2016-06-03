from ..distributed_helper import get_cluster_spec
import tensorflow as tf
import sys

FLAGS = tf.app.flags.FLAGS

def fc_distributed_data_ff():
    batch_size = int(FLAGS.batch_size / FLAGS.num_workers)
    with tf.device(tf.train.replica_device_setter(cluster=get_cluster_spec())):
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Define model
        x = tf.random_uniform((batch_size, FLAGS.weight_size),
                               minval=-1, maxval=1)
        for i in range(FLAGS.level):
            W = tf.Variable(tf.zeros([int(x.get_shape()[1]), FLAGS.weight_size]))
            x = tf.matmul(x, W)
        y = tf.nn.softmax(x)

        # Define loss and optimizer
        y = tf.reduce_sum(y, reduction_indices=[1])
        y_ = tf.random_uniform([1, batch_size], minval=-1, maxval=1)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                       reduction_indices=[1]))
        return cross_entropy, global_step


def get_run_op():
    assert FLAGS.distributed_mode == 1
    cross_entropy, global_step = fc_distributed_data_ff()
    train_opt = tf.train.GradientDescentOptimizer(0.5)

    # Optimize the variable(s) on "ps" server(s).
    opt = tf.train.SyncReplicasOptimizer(
                        train_opt,
                        replicas_to_aggregate=FLAGS.num_workers,
                        total_num_replicas=FLAGS.num_workers,
                        replica_id=FLAGS.worker_index,
                        name="sync_replicas")
    train_step = opt.minimize(cross_entropy,
                              global_step=global_step)

    # Tensorflow settings
    is_chief = FLAGS.worker_index == 0
    if is_chief:
        chief_queue_runner = opt.get_chief_queue_runner()
        init_tokens_op = opt.get_init_tokens_op()

    init_op = tf.initialize_all_variables()
    sv = tf.train.Supervisor(is_chief=is_chief,
                             init_op=init_op,
                             recovery_wait_secs=1,
                             global_step=global_step)
    sess = sv.prepare_or_wait_for_session(FLAGS.worker_grpc_url)
    print("Worker %d: Session initialization complete." % FLAGS.worker_index)
    if is_chief:
        sv.start_queue_runners(sess, [chief_queue_runner])
        sess.run(init_tokens_op)

    return sess, [train_step, global_step]
