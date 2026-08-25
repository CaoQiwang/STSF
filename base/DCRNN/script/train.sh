#!/usr/bin/env bash

python base/DCRNN/script/train.py \
  --dataset PEMS-BAY \
  --data-root data \
  --feature 0 \
  --seq-len 12 \
  --horizon 12 \
  --batch-size 64 \
  --epochs 100 \
  --lr 0.01 \
  --epsilon 0.001 \
  --rnn-units 64 \
  --rnn-layers 2 \
  --max-diffusion-step 2 \
  --cl-decay-steps 2000 \
  --max-grad-norm 5 \
  --patience 50 \
  --lr-milestones 20 30 40 50 \
  --lr-decay-ratio 0.1 \
  --device auto
