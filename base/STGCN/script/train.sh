#!/usr/bin/env bash
# bash base/STGCN/script/train.sh

python base/STGCN/script/train.py \
  --dataset PEMS-BAY \
  --data-root data \
  --feature 0 \
  --n-his 12 \
  --n-pred 9 \
  --ks 3 \
  --kt 3 \
  --batch-size 50 \
  --epochs 50 \
  --lr 0.001 \
  --device auto
