#!/bin/bash

dataset=$1
structure=$2
portion=$3
. .venv/bin/activate
for i in 0 1 2 3; do
	./run_sweep.sh $i ${dataset} ${structure} ${portion}&
done
deactivate
