#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name=mnist_freeze
#SBATCH --output="logs/%j_mnist_freeze.txt"
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
module load python/3.10
source .venv/bin/activate
cd /home/mila/k/kusha.sareen/molecule_generation/MNISTDiffusion/
unset CUDA_VISIBLE_DEVICES
python train_mnist.py --exp_name mnist_freeze --use_rot_mnist --use_canon --num_rot 16 --freeze
