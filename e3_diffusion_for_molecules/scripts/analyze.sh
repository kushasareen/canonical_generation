#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name=analyze
#SBATCH --output="logs/%j_analyze.txt"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
module load python/3.10
source .venv/bin/activate
cd /home/mila/k/kusha.sareen/molecule_generation/e3_diffusion_for_molecules/
unset CUDA_VISIBLE_DEVICES
python eval_analyze.py --model_path outputs/canon_nf64_nl9_lr4e-4_unf_resume --n_samples 1000