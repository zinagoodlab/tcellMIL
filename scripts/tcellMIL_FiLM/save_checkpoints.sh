#!/bin/bash
#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/kcytsui/24c_save_ckpt_%j.out
#SBATCH --error=/scratch/users/kcytsui/24c_save_ckpt_%j.err
#SBATCH --job-name=24c_save_ckpt

export PYTHONNOUSERSITE=1
export WANDB_MODE=offline

eval "$(conda shell.bash hook)"
conda activate /scratch/users/kcytsui/conda-envs/scgpt_h100

mkdir -p /home/users/kcytsui/tcellMIL/results/cohort_aware_antifit_ckpt/
cd /home/users/kcytsui/tcellMIL/results/cohort_aware_antifit_ckpt/
python /home/users/kcytsui/tcellMIL/scripts/tcellMIL_FiLM/save_checkpoints.py
