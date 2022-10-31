#!/bin/bash

# B
echo "hi"
for ((i=50; i<=120; i=i+1)); do
    python3 train2-1_B_WGAN.py $1 --mode="test" --pth_name="G_${i}.pth"
    python -m pytorch_fid $1 "./hw2_data/face/val"
    python3 face_recog.py --image_dir=$1
done

# Usage: bash ./eval2-1B.sh ./output_images >> ./log/train2-1_tutorial_B
