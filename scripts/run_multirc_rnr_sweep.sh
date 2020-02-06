#!/bin/bash

lambdas=( 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1. 2. 5. )
gpu_id=0
batch_size=16
num_epochs=10
dataset='multirc'
exp_structure='rnr'
benchmark_split='val'
train_on_portion='0.4'

for par_lambda in ${lambdas[@]}; do
	python bert_as_tfkeras_layer.py --par_lambda ${par_lambda} --gpu_id ${gpu_id} --batch_size ${batch_size} --num_epochs ${num_epochs} --dataset ${dataset} --evaluate --exp_benchmark --exp_structure ${exp_structure} --merge_evidences --benchmark_split ${benchmark_split} --do_train --delete_checkpoints --train_on_portion ${train_on_portion};
done
