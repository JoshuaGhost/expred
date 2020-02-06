#!/bin/bash

par_lambda=$1
gpu_id=$2
batch_size=16
datasets=$3
num_epochs=10
exp_structure=$4
benchmark_split='test'
train_on_portion='0'

for dataset in ${datasets[@]}; do
	python bert_as_tfkeras_layer.py --par_lambda ${par_lambda} --gpu_id ${gpu_id} --batch_size ${batch_size} --num_epochs ${num_epochs} --dataset ${dataset} --exp_structure ${exp_structure} --merge_evidences --exp_visualize;
done
