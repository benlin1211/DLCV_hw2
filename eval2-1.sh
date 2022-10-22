#!/bin/bash

python -m pytorch_fid $1 "./hw2_data/face/val"
python3 face_recog.py --image_dir=$1

# Usage: bash ./eval2-1.sh ./output_images
