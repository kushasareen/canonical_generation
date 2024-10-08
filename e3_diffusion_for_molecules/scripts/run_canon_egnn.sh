#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=canon_diffusion_egnn
#SBATCH --output=logs/%j_out.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
module load python/3.10
source .venv/bin/activate
cd /home/mila/k/kusha.sareen/molecule_generation/e3_diffusion_for_molecules/
unset CUDA_VISIBLE_DEVICES
python main_qm9.py --n_epochs 3000 --exp_name edm_qm9_canon_egnn --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --use_canon --model egnn_dynamics --break_sym --canon_n_layers 5 --canon_nf 64