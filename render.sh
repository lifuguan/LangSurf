#!/bin/bash

# base_path="data/scannet/scene0085_00"
base_path=$1
dataset_path=$base_path
dataset_name=$(basename $base_path)
casename=$(basename $base_path)

for level in 1 2 3
do
    echo ${base_path}/output/${casename}_${level}
    python render.py -m ${base_path}/output/${casename}_${level} --include_feature --normalized
done