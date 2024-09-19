ckpt=$1

python3 experiments/joint_training/assess_distribution_learning.py \
        --path_to_model_ckpt ${ckpt} \
        --path_to_model_config configs/models/jointformer_separate_task_token \
        --path_to_tokenizer_config configs/tokenizers/smiles_separate_task_token \
        --chembl_training_file ../jf-data/data/guacamol/train/smiles.txt \
        --output outputs/jointformer_separate_task_token/assess_distribution_learning.json

#        --chembl_training_file ../jf-data/data/guacamol/val/smiles.txt \