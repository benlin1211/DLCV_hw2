#!/bin/bash

# TODO - run your inference Python3 code


# https://stackoverflow.com/questions/229551/how-to-check-if-a-string-contains-a-substring-in-bash
if [[ $1 == *"usps"* ]]; then
    echo "usps dataset"
    echo $1 $2 
    python3 inference2-3_usps.py $1 $2
elif [[ $1 == *"svhn"* ]]; then
    echo "svhn dataset"
    echo $1 $2 
    python3 inference2-3_svhn.py $1 $2 
fi

# bash hw2_3.sh  "./hw2_data_mock/digits/svhn/val" "./test_pred_svhn.csv"
# bash hw2_3.sh  "./hw2_data_mock/digits/usps/val" "./test_pred_usps.csv"