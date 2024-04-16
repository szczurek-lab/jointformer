#!/bin/bash

#python ./scripts/pretrain/train.py \
#    --out_dir /raid/aizd/hybrid_transformer/results/pre-train/guacamol_no_overfit/jointformer \
#    --path_to_task_config ./configs/tasks/guacamol/distribution_learning/config.json \

python ./scripts/pretrain/eval.py \
    --out_dir /raid/aizd/hybrid_transformer/results/pretrain/hybrid_transformer \
    --path_to_task_config ./configs/tasks/guacamol/distribution_learning/config.json \
    --temperature 1.2 \
    --top_k 10

#python ./scripts/pretrain/eval.py \
#    --out_dir /raid/aizd/hybrid_transformer/results/pre-train/guacamol_no_overfit/jointformer \
#    --path_to_task_config ./configs/tasks/guacamol/distribution_learning/config.json \
