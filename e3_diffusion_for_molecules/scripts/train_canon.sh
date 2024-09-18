#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name=canon2
#SBATCH --output="logs/%j_canon2.txt"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
module load python/3.10
source .venv/bin/activate
cd /home/mila/k/kusha.sareen/molecule_generation/e3_diffusion_for_molecules/
unset CUDA_VISIBLE_DEVICES
exit_script() {

    echo "Preemption signal, saving myself"
    trap - SIGTERM # clear the trap
    bash scripts/resume.sh -n canon2 -r 1e-4
    kill -- -$$

}
trap exit_script SIGTERM

python main_qm9.py --n_epochs 3000 --exp_name "canon2" --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --use_canon --model gnn_dynamics --canon_n_layers 9 --canon_nf 64
