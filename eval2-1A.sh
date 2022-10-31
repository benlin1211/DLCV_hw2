#!/bin/bash

# A
for ((i=1; i<=400; i=i+1)); do
    python3 train2-1_tutorial.py $1 --mode="test" --pth_name="G_${i}.pth"
    python -m pytorch_fid $1 "./hw2_data/face/val"
    python3 face_recog.py --image_dir=$1
done

# Usage: bash ./eval2-1A.sh ./output_images >> ./log/train2-1_tutorial_A