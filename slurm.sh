#!/bin/bash
#SBATCH -p normal
#SBATCH --time=230:59:59
#SBATCH --account=root
#SBATCH --job-name=irem
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/baytas/Documents/Irem/tez_v2_clean/Outputs/out-%j.out
#SBATCH --error=/home/baytas/Documents/Irem/tez_v2_clean/Errors/err-%j.err

module load Python/3.12.0 cuda/10.1
 
python3.12 train.py
