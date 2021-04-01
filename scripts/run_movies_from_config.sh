#!/bin/bash

dataset_path=~/datasets/eraser
dataset=movies # movies, fever, multirc

output_dir=outputs/${dataset}/`date +"%y_%d_%m_%H_%M_%S"`
data_dir=${dataset_path}/${dataset}

export PYTHONPATH=$PYTHONPATH:./ && python expred/train.py --data_dir ${data_dir} --output_dir ${output_dir} --conf "./params/${dataset}_expred.json" --batch_size 4 
