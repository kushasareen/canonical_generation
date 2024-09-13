#!/bin/bash
while getopts n:e:r: flag
do
    case "${flag}" in
        n) name=${OPTARG};;
        e) epoch=${OPTARG};;
        r) lr=${OPTARG};;

    esac
done

sbatch <<EOT
#!/bin/bash

#SBATCH --partition=long
#SBATCH --job-name="${name}_resume"
#SBATCH --output="logs/%j_${name}_resume.txt"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
module load python/3.10
source .venv/bin/activate
cd /home/mila/k/kusha.sareen/molecule_generation/e3_diffusion_for_molecules/
unset CUDA_VISIBLE_DEVICES
python main_qm9.py --resume "outputs/${name}" --start_epoch ${epoch} --lr ${lr} --exp_name $name
EOT