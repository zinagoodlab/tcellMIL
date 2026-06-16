#!/bin/bash
#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/kcytsui/27_perturbation_%j.out
#SBATCH --error=/scratch/users/kcytsui/27_perturbation_%j.err
#SBATCH --job-name=27_pert

export PYTHONNOUSERSITE=1
export WANDB_MODE=offline

eval "$(conda shell.bash hook)"
conda activate /scratch/users/kcytsui/conda-envs/scgpt_h100

mkdir -p /home/users/kcytsui/tcellMIL/results/perturbation_24c/
cd /home/users/kcytsui/tcellMIL/scripts/tcellMIL_FiLM/
python /home/users/kcytsui/tcellMIL/scripts/tcellMIL_FiLM/insilico_perturbation.py \
    --ckpt_output_dir /home/users/kcytsui/tcellMIL/results/cohort_aware_antifit_ckpt/ \
    --out_dir /home/users/kcytsui/tcellMIL/results/perturbation_24c/ \
    --mad_multiplier 100
