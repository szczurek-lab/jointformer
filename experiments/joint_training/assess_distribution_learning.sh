model=jointformer_separate_task_token
ckpt=checkpoints/jointformer_separate_task_token/ckpt.pt

temps=(0.8 1.0)
topks=(25 25)

for i in {0..1}; do
        temp=${temps[$i]}
        topk=${topks[$i]}
        echo $i $temp $topk
        python3 experiments/joint_training/assess_distribution_learning.py \
                --path_to_model_ckpt ${ckpt} \
                --path_to_model_config configs/models/${model} \
                --path_to_tokenizer_config configs/tokenizers/smiles_separate_task_token \
                --chembl_training_file data/guacamol/train/smiles.txt \
                --output outputs/${model}/generate_${temp}_${topk}/assess_distribution_learning.js \
                --temperature $temp \
                --top_k $topk \
                --batch_size 256
done