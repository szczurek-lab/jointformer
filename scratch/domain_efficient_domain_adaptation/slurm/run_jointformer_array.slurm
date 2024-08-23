#!/bin/bash

#SBATCH --job-name=jointformer-domain-efficient 
#SBATCH --array=0-5    
#SBATCH --output jointformer-domain-efficient_%a.log                              
#SBATCH --error jointformer-domain-efficient_%a.log                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_long

#SBATCH --nodes=1                
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=4   
#SBATCH --gres=gpu:1                                    
#SBATCH --mem=10G                                                      
#SBATCH --time=96:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=99                                        
#SBATCH --reservation=res_adam.izdebski                      

datasets=("esol" "freesolv" "lipo")
splitters=("random" "scaffold")

# Create a list of combinations
combinations=()
for dataset in "${datasets[@]}"; do
  for splitter in "${splitters[@]}"; do
    combinations+=("$dataset,$splitter")
  done
done

# Get the specific combination for this job
combination=${combinations[$SLURM_ARRAY_TASK_ID]}
dataset=$(echo $combination | cut -d',' -f1)
splitter=$(echo $combination | cut -d',' -f2)

# Run the script
echo "Running $dataset $splitter split..."

source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments


python3 experiments/data_efficient_domain_adaptation/train.py \
   --data_dir /lustre/groups/aih/jointformer/data \
   --out_dir /lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation/molecule_net/$splitter/$dataset/jointformer/ \
   --path_to_dataset_config configs/datasets/molecule_net/$splitter/$dataset \
   --path_to_tokenizer_config configs/tokenizers/smiles \
   --path_to_model_config configs/models/jointformer \
   --path_to_trainer_config configs/trainers/finetune \
   --path_to_model_ckpt /lustre/groups/aih/jointformer/results/pre-train/lm_psychem/no_separate_token/cls_emb/bs_512/drop_0/seed_1337/ckpt.pt \


python3 experiments/data_efficient_domain_adaptation/test.py \
   --data_dir /lustre/groups/aih/jointformer/data \
   --out_dir /lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation/molecule_net/$splitter/$dataset/jointformer/ \
   --path_to_dataset_config configs/datasets/molecule_net/$splitter/$dataset \
   --path_to_tokenizer_config configs/tokenizers/smiles \
   --path_to_model_config configs/models/jointformer \
   --path_to_trainer_config configs/trainers/finetune \
