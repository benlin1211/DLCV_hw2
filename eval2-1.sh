#!/bin/bash

python -m pytorch_fid "./output_images/" "./hw2_data/face/val"
python3 face_recog.py --image_dir="./output_images/"
