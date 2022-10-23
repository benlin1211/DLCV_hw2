#!/bin/bash

#python3 train2-1_A.py $1 --mode="train"
python3 train2-1_B_wandb.py ./output_images --mode="train"

# bash ./train2-1_B_wandb.sh ./output_images