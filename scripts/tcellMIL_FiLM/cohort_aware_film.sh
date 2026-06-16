#!/bin/bash
#SBATCH --account=zinaida
#SBATCH --partition=cmackall
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=cohort_aware_antifit_%j.out
#SBATCH --error=cohort_aware_antifit_%j.err
#SBATCH --job-name=cohort_aware_antifit

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook)"
conda activate /scratch/users/kcytsui/conda-envs/scgpt_h100

cd /home/users/kcytsui/tcellMIL/results/cohort_aware_antifit/
python /home/users/kcytsui/tcellMIL/scripts/tcellMIL_FiLM/cohort_aware_film.py
