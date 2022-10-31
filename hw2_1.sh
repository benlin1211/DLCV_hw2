#!/bin/bash

# A
python3 train2-1_A.py $1 --mode="test" --pth_name="G_277.pth"
# B
#python3 train2-1_B.py $1 --mode="test"
# python3 train2-1_B_wandb.py $1 --mode="test"

# Usage: bash ./hw2_1.sh ./output_images
# python -m pytorch_fid "./output_images" "./hw2_data/face/val"
# python3 face_recog.py --image_dir="./output_images"