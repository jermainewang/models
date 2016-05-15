# Mini-benchmark for Tensorflow

Typical usage:
python run_mini.py --num_workers=WORKER_COUNT --level=LEVEL_COUNT --mode=MODEL_NAME --weight_size=WEIGHT_SIZE --batch_size=BATCH_SIZE --num_batch=LOOP_COUNT

Mode list:
1. Fully connected FF, single GPU (WORKER_COUNT = 1, mode=fc_data_ff)
2. Fully connected FF, multiple GPU, data parallelism  (WORKER_COUNT > 1, mode=fc_data_ff)
3. Fully connected FF, multiple GPU, model parallelism  (WORKER_COUNT > 1, mode=fc_model_ff)
4. Fully connected, single GPU (WORKER_COUNT = 1, mode=fc_data)
5. Fully connected, multiple GPU, data parallelism  (WORKER_COUNT > 1, mode=fc_data)
6. Fully connected, multiple GPU, model parallelism  (WORKER_COUNT > 1, mode=fc_model)
7. Fully connected, single GPU, manual BP (WORKER_COUNT = 1, mode=fc_single_manual)
8. Fully connected, multiple GPU,manual BP,  data parallelism  (WORKER_COUNT > 1, mode=fc_data_manual)
9. Fully connected, multiple GPU, manual BP, model parallelism  (WORKER_COUNT > 1, mode=fc_model_manual)
