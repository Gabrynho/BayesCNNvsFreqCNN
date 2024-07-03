#!/bin/bash
#SBATCH -A dssc
#SBATCH -p GPU
#SBATCH --job-name=PML_training
#SBATCH --nodes=1
#SBATCH --mem=32gb
#SBATCH --time=2:0:0
#SBATCH --gpus=1

source /u/dssc/gmasella/jupyter/bin/activate
python3 /u/dssc/gmasella/BayesCNNvsFreqCNN/f10_training_mislabel.py