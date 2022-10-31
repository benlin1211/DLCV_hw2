#!/bin/bash

# A
for ((i=200; i<=300; i=i+1)); do
    python3 train2-1_A.py $1 --mode="test" --pth_name="G_${i}.pth"
    python -m pytorch_fid $1 "./hw2_data/face/val"
    python3 face_recog.py --image_dir=$1
done

# python3 train2-1_A.py $1 --mode="test" --pth_name="G_49.pth"
# python -m pytorch_fid $1 "./hw2_data/face/val"
# python3 face_recog.py --image_dir=$1

# Usage: bash ./eval2-1.sh ./output_images >> ./log/train2-1.txt
