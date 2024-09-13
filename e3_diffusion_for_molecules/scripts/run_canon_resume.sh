#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=canon
#SBATCH --output=logs/%j_canon.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
module load python/3.10
source .venv/bin/activate
cd /home/mila/k/kusha.sareen/molecule_generation/e3_diffusion_for_molecules/
unset CUDA_VISIBLE_DEVICES
python main_qm9.py --resume outputs/edm_qm9_canon_resume --start_epoch 80 --exp_name edm_qm9_canon_resume