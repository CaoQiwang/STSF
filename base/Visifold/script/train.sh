#!/usr/bin/env bash
set -euo pipefail

for dataset in PEMS04 PEMS08; do
  for horizon in 24 36 48; do
    python script/train.py --dataset "$dataset" --in-steps "$horizon" --out-steps "$horizon"
  done
done
