#!/bin/bash

#SBATCH --job-name=jointformer-training                              
#SBATCH --output logs/output-%x.log                              
#SBATCH --error logs/error-%x.log                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_long

#SBATCH --nodes=1                                           
#SBATCH --ntasks-per-node=1                                 
#SBATCH --gpus-per-task=1                                   
#SBATCH --cpus-per-task=8                                   
#SBATCH --mem=20G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=48:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=maximilian.armuss@helmholtz-munich.de

#SBATCH --nice=1000                              

conda run python -m pip install -r requirements.txt

srun sh experiments/joint_training/assess_distribution_learning_maxi.sh ~/results/ckpt.pt ~/results/data/data/guacamol/train/smiles.txt
