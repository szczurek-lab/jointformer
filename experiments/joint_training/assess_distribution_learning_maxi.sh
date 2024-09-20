ckpt=$1
dataset=$2

python3 experiments/joint_training/assess_distribution_learning.py \
        --path_to_model_ckpt ${ckpt} \
        --path_to_model_config configs/models/jointformer_separate_task_token \
        --path_to_tokenizer_config configs/tokenizers/smiles_separate_task_token \
        --chembl_training_file ${dataset} \
        --output outputs/jointformer_separate_task_token/assess_distribution_learning.json \
        > outputs/jointformer_separate_task_token/stdout.log \
        2> outputs/jointformer_separate_task_token/stderr.log
