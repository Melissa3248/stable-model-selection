#!/bin/bash
#SBATCH -G1
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --job-name=reg_0
#SBATCH --output=temp_job/n0.out
python compute_coefficients.py --dataset_idx 0 --solver 'lasso' --data_filepath 'data/regression_rho0.99' --results_filepath 'results/regression_rho0.99' --J 30