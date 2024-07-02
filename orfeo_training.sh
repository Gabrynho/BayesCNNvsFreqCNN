#!/bin/bash
#SBATCH -A dssc
#SBATCH -p GPU
#SBATCH --job-name=PML_training
#SBATCH --nodes=1
#SBATCH --tasks=24 
#SBATCH --mem=128gb 
#SBATCH --time=2:0:0
#SBATCH --gpus=1

source jupyter/bin/activate
python3 import torch as th
python3 th.cuda.is_available() 
#python3 c10_training_lr0.001.py