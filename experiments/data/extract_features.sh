splits=(train test val)

model=$1
ckpt=$2
tokenizer=$3

for split in ${splits[@]}; do

python3 experiments/data/extract_features.py \
        --path_to_model_ckpt ${ckpt} \
        --path_to_model_config configs/models/${model} \
        --path_to_dataset_config configs/datasets/guacamol/unsupervised \
        --path_to_tokenizer_config configs/tokenizers/${tokenizer} \
        --split ${split} \
        --output outputs/${model}/${split}/guacamol_v1_features.npy \
        --chunk_size 30000

done
