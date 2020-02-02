#!/bin/bash

lambdas=( 1. ) 
gpu_id=0
train_first=( exp cls )
batch_size=16
num_epochs=2
dataset='movies'
exp_structure='rnr'
benchmark_split='test'
train_on_portion='0.1'

for phase in ${train_first[@]}; do
	for par_lambda in ${lambdas[@]}; do
		python bert_as_tfkeras_layer.py --par_lambda ${par_lambda} --gpu_id ${gpu_id} --batch_size ${batch_size} --num_epochs ${num_epochs} --dataset ${dataset} --evaluate --exp_benchmark --exp_structure ${exp_structure} --merge_evidences --benchmark_split ${benchmark_split} --do_train --delete_checkpoints --train_on_portion ${train_on_portion} --train_${phase}_first;
	done
done
