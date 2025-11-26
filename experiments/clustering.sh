#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --job-name=clustering
#SBATCH --output=temp_job/clustering.out

# argmax A 
python subset_selections_kmeans.py --selection_method 'argmax' --config_option 'kmeans' --results_filepath 'results/clustering_elbow' --J 30

# top 1 (argmax of bootstrapped A)
python subset_selections_kmeans.py --selection_method 'top_k' --k 1 --config_option 'bootstrap_kmeans' --results_filepath 'results/clustering_elbow' --J 30 

# top 2
python subset_selections_kmeans.py --selection_method 'top_k' --k 2 --config_option 'bootstrap_kmeans' --results_filepath 'results/clustering_elbow' --J 30

# inflated argmax 
python subset_selections_kmeans.py --selection_method 'inflated_argmax' --epsilon 0.01 --config_option 'bootstrap_kmeans' --results_filepath 'results/clustering_elbow' --J 30
