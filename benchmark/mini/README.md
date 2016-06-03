## Mini-benchmark for Tensorflow

### Usage of the benchmark:
python run_mini.py --num_workers=WORKER_COUNT --level=LEVEL_COUNT --mode=MODEL_NAME --weight_size=WEIGHT_SIZE --batch_size=BATCH_SIZE --num_batch=LOOP_COUNT

### Mode list:
1.  Fully connected FF, single GPU (WORKER_COUNT = 1, MODEL_NAME=fc_data_ff)
2.  Fully connected FF, multiple GPU, data parallelism (WORKER_COUNT > 1, MODEL_NAME = fc_data_ff)
3.  Fully connected FF, multiple GPU, model parallelism (WORKER_COUNT > 1, MODEL_NAME = fc_model_ff)
4.  Fully connected, single GPU (WORKER_COUNT = 1, MODEL_NAME = fc_data)
5.  Fully connected, multiple GPU, data parallelism (WORKER_COUNT > 1, MODEL_NAME = fc_data)
6.  Fully connected, multiple GPU, model parallelism (WORKER_COUNT > 1, MODEL_NAME = fc_model)
7.  Fully connected, single GPU, manual BP (WORKER_COUNT = 1, MODEL_NAME = fc_single_manual)
8.  Fully connected, multiple GPU,manual BP, data parallelism (WORKER_COUNT > 1, MODEL_NAME = fc_data_manual)
9.  Fully connected, multiple GPU, manual BP, model parallelism (WORKER_COUNT > 1, MODEL_NAME = fc_model_manual)
10. A model to simulate physics usage.  (MODEL_NAME = fc_phy, --phy_blocks=2(or more) --shared_ratio=0.8 (depend on the usage))

## Log (Tensorboard):
### Usage of the benchmark
1. Add these two parameters : --full_trace=1 --log_dir="YourCustomizedDirectory"
2. After execution, change to the tensorflow directory.
3. python tensorflow/tensorboard/tensorboard.py  --logdir="YourCustomizedDirectory"

### Source Code
Add following code before session.run().
```
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',
				  session.graph)
test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
```

## Trace
### Usage of the benchmark
1. Add the parameter : --full_trace=1
2. After execution, there will be a trace file, full_trace.ctf, in the execution directory.
3. Download full_trace.ctf to your personal computer.
4. git clone https://github.com/catapult-project/catapult
5. catapult/tracing/bin/trace2html full_trace.ctf --output=my_trace.html
6. open my_trace.html

### Source Code
Before session.run()
```
    from tensorflow.core.protobuf import config_pb2
    run_options = config_pb2.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
```
session.run()
```
    session.run(your_arguments, options=run_options, run_metadata=run_metadata)
```
After session.run()
```
    tl = tf.python.client.timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('full_trace.ctf', 'w') as f:
	f.write(ctf)
```

