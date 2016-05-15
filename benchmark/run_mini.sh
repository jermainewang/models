#!/bin/bash

for ws in 1024 2048 4096 8192
do
    for bs in 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
    do
        python run_mini.py --num_workers=1 --level=10 --mode=fc_data --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 >> result
        python run_mini.py --num_workers=2 --level=10 --mode=fc_data --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 --gpu_aggr=0 >> result
        python run_mini.py --num_workers=2 --level=10 --mode=fc_data --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 --gpu_aggr=1 >> result
        python run_mini.py --num_workers=2 --level=10 --mode=fc_data_slice --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 --gpu_aggr=0 >> result
        python run_mini.py --num_workers=2 --level=10 --mode=fc_data_slice --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 --gpu_aggr=1 >> result
        python run_mini.py --num_workers=2 --level=10 --mode=fc_model --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 >> result
        python run_mini.py --num_workers=1 --level=10 --mode=fc_data_ff --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 >> result
        python run_mini.py --num_workers=2 --level=10 --mode=fc_data_ff --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 --gpu_aggr=0 >> result
        python run_mini.py --num_workers=2 --level=10 --mode=fc_data_ff --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 --gpu_aggr=1 >> result
        python run_mini.py --num_workers=2 --level=10 --mode=fc_model_ff --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 >> result
        python run_mini.py --num_workers=1 --level=10 --mode=fc_single_manual --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 --no_assign=0 >> result
        python run_mini.py --num_workers=2 --level=10 --mode=fc_model_manual --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 --no_assign=0 >> result
        python run_mini.py --num_workers=2 --level=10 --mode=fc_data_manual --weight_size=$ws --batch_size=$bs --num_batches=200 --num_classes=1 --no_assign=0 >> result
    done
done
