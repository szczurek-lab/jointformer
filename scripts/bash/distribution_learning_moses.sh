#!/bin/bash
#
#python ./scripts/pretrain/train.py \
#    --out_dir /raid/aizd/hybrid_transformer/results/pre-train/moses/jointformer \
#    --path_to_task_config ./configs/tasks/moses/distribution_learning/config.json \
#
python ./scripts/pretrain/eval.py \
    --out_dir /raid/aizd/hybrid_transformer/results/pre-train/moses/jointformer \
    --path_to_task_config ./configs/tasks/moses/distribution_learning/config.json \
