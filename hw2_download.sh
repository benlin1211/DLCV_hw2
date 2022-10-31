#!/bin/bash

python3 download_model.py
# https://github.com/wkentaro/gdown/issues/163
unzip -o ckpt2-1A.zip 
unzip -o ckpt2-2.zip 
unzip -o ckpt2-3_svhn.zip
unzip -o ckpt2-3_usps.zip
