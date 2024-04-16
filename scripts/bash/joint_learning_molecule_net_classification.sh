#!/bin/bash

#python ./scripts/joint_learning/train.py \
#    --out_dir /raid/aizd/hybrid_transformer/results_v10/joint_learning/molecule_net_classification \
#    --benchmark molecule_net_classification \
#    --model HybridTransformer \
#    --task_p 0.1

python ./scripts/joint_learning/eval.py \
    --out_dir /raid/aizd/hybrid_transformer/results_v10/joint_learning/molecule_net_classification \
    --benchmark molecule_net_classification \
    --model HybridTransformer \
    --task_p 0.1
