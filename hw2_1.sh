#!/bin/bash

# A
# python3 train2-1_tutorial.py $1 --mode="test"
python3 train2-1_A.py $1 --mode="test" --pth_name="G_92.pth"
# python3 train2-1_Aplus_wandb.py $1 --mode="test"

# B
#python3 train2-1_B.py $1 --mode="test"
# python3 train2-1_B_wandb.py $1 --mode="test"
# Usage: bash ./hw2_1.sh ./output_images
