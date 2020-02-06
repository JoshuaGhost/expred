#!/bin/bash

par_lambda=1.
#gpu_id=0
#train_first=( exp cls )
gpu_id=$1
phase=$2
#batch_size=16
batch_size=$3
num_epochs=10
datasets=( fever multirc )
exp_structure='gru'
benchmark_split='test'
train_on_portion='0'

for dataset in ${datasets[@]}; do
	python bert_as_tfkeras_layer.py --par_lambda ${par_lambda} --gpu_id ${gpu_id} --batch_size ${batch_size} --num_epochs ${num_epochs} --dataset ${dataset} --evaluate --exp_benchmark --exp_structure ${exp_structure} --load_phase1 --merge_evidences --benchmark_split ${benchmark_split} --do_train --start_from_phase1 --train_on_portion ${train_on_portion} --train_${phase}_first;
done
