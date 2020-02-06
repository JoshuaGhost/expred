#!/bin/bash

gpu_id=$1
dataset=$2
exp_structure=$3
train_on_portion=$4
if [[ $exp_structure == 'gru' ]]; then
	l0='0.1,0.2,0.5'
	l1='1,2,5'
	l2='10,20,50'
	l3='100,200,500'
else
	l0='0.001,0.002,0.005'
	l1='0.01,0.02,0.05'
	l2='0.1,0.2,0.5'
	l3='1.,2.,5'
fi
lambdas=( $l0 $l1 $l2 $l3 )
IFS=',' read -r -a lambdas<<<${lambdas[$gpu_id]}
batch_size=4
num_epochs=10
benchmark_split='val'

for par_lambda in ${lambdas[@]}; do
	python bert_as_tfkeras_layer.py --par_lambda ${par_lambda} --gpu_id ${gpu_id} --batch_size ${batch_size} --num_epochs ${num_epochs} --dataset ${dataset} --evaluate --exp_benchmark --exp_structure ${exp_structure} --merge_evidences --benchmark_split ${benchmark_split} --do_train --delete_checkpoints --train_on_portion ${train_on_portion};
done
