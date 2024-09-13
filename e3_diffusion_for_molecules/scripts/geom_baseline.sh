#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name=geom_baseline
#SBATCH --output="logs/%j_geom_baseline.txt"
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=167:00:00
module load python/3.10
source .venv/bin/activate
cd /home/mila/k/kusha.sareen/molecule_generation/e3_diffusion_for_molecules/
unset CUDA_VISIBLE_DEVICES
python main_geom_drugs.py --n_epochs 3000 --exp_name geom_baseline --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 16 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --visualize_every_batch 10000
