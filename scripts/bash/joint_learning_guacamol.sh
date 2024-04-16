#!/bin/bash

python ./scripts/joint_learning/train.py \
    --out_dir /raid/aizd/hybrid_transformer/results_rebuttal/joint_learning/guacamol \
    --benchmark guacamol \
    --model HybridTransformer

python ./scripts/joint_learning/eval.py \
    --out_dir /raid/aizd/hybrid_transformer/results_rebuttal/joint_learning/guacamol \
    --benchmark guacamol \
    --model HybridTransformer
