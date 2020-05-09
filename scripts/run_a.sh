#!/bin/bash

lambdas=( 10. 20. 50. )
gpu_id=0
batch_size=16
num_epochs=10
dataset='movies'
exp_structure='rnr'
benchmark_split='test'

for par_lambda in ${lambdas[@]}; do
	python bert_as_tfkeras_layer.py --par_lambda ${par_lambda} --gpu_id ${gpu_id} --batch_size ${batch_size} --num_epochs ${num_epochs} --dataset ${dataset} --evaluate --exp_benchmark --exp_structure ${exp_structure} --merge_evidences --benchmark_split ${benchmark_split} --do_train --delete_checkpoints;
done

#lambdas=( 2 )
#gpu_id=1
#dataset='fever'

#for par_lambda in ${lambdas[@]}; do
#	python learn_to_interpret.py --par_lambda ${par_lambda} --gpu_id ${gpu_id} --batch_size ${batch_size} --num_epochs ${num_epochs} --dataset ${dataset} --evaluate --exp_benchmark --exp_structure ${exp_structure} --merge_evidences --benchmark_split ${benchmark_split};
#done

#lambdas=( 2 )
#gpu_id=1
#dataset='multirc'

#for par_lambda in ${lambdas[@]}; do
#	python learn_to_interpret.py --par_lambda ${par_lambda} --gpu_id ${gpu_id} --batch_size ${batch_size} --num_epochs ${num_epochs} --dataset ${dataset} --evaluate --exp_benchmark --exp_structure ${exp_structure} --merge_evidences --benchmark_split ${benchmark_split};
#done
