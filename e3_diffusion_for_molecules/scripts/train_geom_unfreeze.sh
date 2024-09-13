#!/bin/bash
while getopts f:l:r: flag
do
    case "${flag}" in
        f) nf=${OPTARG};;
        l) nl=${OPTARG};;
        r) lr=${OPTARG};;

    esac
done

NAME="geom_nf${nf}_nl${nl}_lr${lr}_unf";
sbatch <<EOT
#!/bin/bash

#SBATCH --partition=main
#SBATCH --job-name=$NAME
#SBATCH --output="logs/%j_${NAME}.txt"
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=119:00:00
module load python/3.10
source .venv/bin/activate
cd /home/mila/k/kusha.sareen/molecule_generation/e3_diffusion_for_molecules/
unset CUDA_VISIBLE_DEVICES
python main_geom_drugs.py --n_epochs 3000 --exp_name $NAME --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 16 --nf 256 --n_layers 4 --lr $lr --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model gnn_dynamics --visualize_every_batch 10000 --use_canon  --canon_n_layers $nl --canon_nf $nf
EOT