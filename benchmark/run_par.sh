#!/bin/bash

for ws in 1024 2048 4096 8192
do
    for bs in 512 1024 2048 4096 8192
    do
        echo "Weight: $ws, Batch: $bs"
        echo "single"
        python modelpar.py --batch_size=$bs --hidden_size=$ws --num_gpus=1 --num_layers=10 --num_batches=50 2>&1 | tail -n 1
        echo "modelx2"
        python modelpar.py --batch_size=$bs --hidden_size=$ws --num_gpus=2 --num_layers=10 --num_batches=50 2>&1 | tail -n 1
        echo "modelx4"
        python modelpar.py --batch_size=$bs --hidden_size=$ws --num_gpus=4 --num_layers=10 --num_batches=50 2>&1 | tail -n 1
        echo "datax2"
        python datapar.py --batch_size=$bs --hidden_size=$ws --num_gpus=2 --num_layers=10 --num_batches=50 2>&1 | tail -n 1
        echo "datax4"
        python datapar.py --batch_size=$bs --hidden_size=$ws --num_gpus=4 --num_layers=10 --num_batches=50 2>&1 | tail -n 1
    done
done
