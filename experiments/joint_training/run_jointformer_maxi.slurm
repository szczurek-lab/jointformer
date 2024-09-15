#!/bin/bash

#SBATCH --job-name=jointformer-training                              
#SBATCH --output output-%x.log                              
#SBATCH --error error-%x.log                                 

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
#SBATCH --constraint=gpu_tesla_A100                        


source $HOME/.bashrc

python -m pip install pipenv
pipenv install .
pipenv run python -m pip install -r requirements.txt

srun pipenv run python experiments/joint_training/train_maxi.py --path_to_repo_dir . --path_to_out_dir ~/results