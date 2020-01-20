#!/bin/bash

lambdas=( 0.001, 0.002, 0.005, 0.02, 0.05 )
gpu_id=0
batch_size=16
num_epochs=10
dataset='movies'
exp_structure='rnr'

for par_lambda in ${lambdas[@]}; do
	python bert_as_tfkeras_layer.py --par_lambda ${par_lambda} --gpu_id ${gpu_id} --batch_size ${batch_size} --num_epochs ${num_epochs} --dataset ${dataset} --do_train --evaluate --exp_benchmark --exp_structure ${exp_structure} --delete_checkpoints --merge_evidences;
done
