#!/bin/bash

# base_path="scannet/scene0114_02"
base_path=$1
dataset_path=$base_path
echo $base_path; echo $(basename $base_path)
dataset_name=$(basename $base_path)
casename=$(basename $base_path)


 # PREPROCESS images for generating semantic featuers
 python preprocess_hcma.py --dataset_path $dataset_path

 cd autoencoder || exit
 # TRAIN the encoder - decoder
 python train.py --dataset_path ../$dataset_path --encoder_dims 512 256 128 64 32 16 3 --decoder_dims 16 32 64 128 256 512 768 --lr 0.0007 --dataset_name $dataset_name
 # INFERENCE the encoded features for langsurf training
 python test.py --dataset_path ../$dataset_path --dataset_name $dataset_name --encoder_dims 512 256 128 64 32 16 3 --decoder_dims 16 32 64 128 256 512 768
 cd ..


for level in 1 2 3
do
    # For scannet dataset, split it into `train` and `val` sets; for lerf-ovs dataset, train all the images
    if [[ "$dataset_path" == *"scannet"* ]]; then
        eval_flag="--eval"
    else
        eval_flag=""
    fi
    python train.py -s $dataset_path -m ${base_path}/output/${casename}  --feature_level ${level}  \
    --max_abs_split_points 0 --opacity_cull_threshold 0.05  --grouping_loss --loss_obj_3d $eval_flag
done

for level in 1 2 3
do
    python render.py -m ${base_path}/output/${casename}_${level} --include_feature --normalized
done



