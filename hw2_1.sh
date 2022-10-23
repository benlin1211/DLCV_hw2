#!/bin/bash

# A
#python3 train2-1_A.py $1 --mode="test"
# python3 train2-1_Aplus_wandb.py $1 --mode="test"

# B
python3 train2-1_B_wandb.py $1 --mode="test"
# bash ./hw2_1.sh ./output_images
