#!/bin/bash
while getopts r: flag
do
    case "${flag}" in
        r) rot=${OPTARG};;
    esac
done

NAME="mnist_rot${rot}_refl";
sbatch <<EOT
#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name=$NAME
#SBATCH --output="logs/%j_${NAME}.txt"
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
module load python/3.10
source .venv/bin/activate
cd /home/mila/k/kusha.sareen/molecule_generation/MNISTDiffusion/
unset CUDA_VISIBLE_DEVICES
python train_mnist.py --exp_name $NAME --use_rot_mnist --use_canon --num_rot $rot --use_refl
EOT