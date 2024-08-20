#!/bin/bash

#SBATCH --job-name=chemberta-domain-efficient
#SBATCH --array=0-5         
#SBATCH --output chemberta-domain-efficient_%a.log                              
#SBATCH --error chemberta-domain-efficient_%a.log                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_long

#SBATCH --nodes=1                                           
#SBATCH --ntasks-per-node=1                                 
#SBATCH --gpus-per-task=1                                   
#SBATCH --cpus-per-task=4                                   
#SBATCH --mem=10G                                           
#SBATCH --gres=gpu:1                                        
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


python3 experiments/data_efficient_domain_adaptation/run.py \
   --data_dir /lustre/groups/aih/jointformer/data \
   --out_dir /lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation/molecule_net/$splitter/$dataset/chemberta/ \
   --path_to_dataset_config configs/datasets/molecule_net/$splitter/$dataset \
   --path_to_tokenizer_config configs/tokenizers/chemberta \
   --path_to_model_config configs/models/chemberta \
   --path_to_trainer_config configs/trainers/finetune \
   --path_to_logger_config configs/loggers/wandb \
