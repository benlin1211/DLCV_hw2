#!/bin/bash


if [[ $1 == *"usps"* ]]; then
    echo "usps dataset"
    python3 eval2-3_personal_use.py $1 $2
elif [[ $1 == *"svhn"* ]]; then
    echo "svhn dataset"
    python3 eval2-3_personal_use.py $1 $2
fi

# Usage: bash eval2-3_personal_use.sh "./hw2_data/digits/svhn/val.csv" "./test_pred_svhn.csv"
# Usage: bash eval2-3_personal_use.sh "./hw2_data/digits/usps/val.csv" "./test_pred_usps.csv"