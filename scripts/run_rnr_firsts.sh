#!/bin/bash

par_lambda=1.
gpu_id=1
#train_first=( exp cls )
train_first=( cls )
batch_size=100
num_epochs=10
datasets=( movies fever multirc )
#datasets=( movies )
exp_structure='rnr'
benchmark_split='test'
train_on_portion='0'

for phase in ${train_first[@]}; do
	for dataset in ${datasets[@]}; do
		python bert_as_tfkeras_layer.py --par_lambda ${par_lambda} --gpu_id ${gpu_id} --batch_size ${batch_size} --num_epochs ${num_epochs} --dataset ${dataset} --evaluate --exp_benchmark --exp_structure ${exp_structure} --merge_evidences --benchmark_split ${benchmark_split} --do_train --start_from_phase1 --train_on_portion ${train_on_portion} --train_${phase}_first;
	done
done
