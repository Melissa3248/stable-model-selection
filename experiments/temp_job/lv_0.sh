#!/bin/bash
#SBATCH -G1
#SBATCH --time=06:00:00
#SBATCH --partition=general
#SBATCH --job-name=lv_0
#SBATCH --output=temp_job/index0.out
python compute_coefficients.py --dataset_idx 0 --solver 'sindy' --data_filepath 'data/lotka_volterra' --results_filepath 'results/lotka_volterra' --J 21