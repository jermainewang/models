import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def get_cluster_spec():
    assert FLAGS.num_workers > 1
    ps_spec = ['geeker-4:2222']
    worker_spec = ['geeker-3:2222', 'geeker-3:2223']
    cluster_spec = tf.train.ClusterSpec({'ps': ps_spec,
                                         'worker': worker_spec})
    return cluster_spec


def get_device_setter(cluster_spec):
    return tf.train.replica_device_setter(cluster=cluster_spec)


CONTINUE=0
NOT_CONTINUE=1
def setup_servers():
    cluster_spec = get_cluster_spec()
    server = tf.train.Server(cluster_spec,
                             job_name=FLAGS.node_name,
                             task_index=FLAGS.worker_index)
    if FLAGS.node_name == 'ps':
        server.join()
        return NOT_CONTINUE
    else:
        return CONTINUE
