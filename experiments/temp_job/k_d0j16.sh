#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --job-name=k_d0j16
#SBATCH --output=temp_job/index0j16.out
python compute_coefficients_kmeans.py --dataset_idx 0 --results_filepath 'results/clustering_elbow' --J 16