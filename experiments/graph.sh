#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=<PARTITION_NAME>
#SBATCH --job-name=graph
#SBATCH --output=temp_job/graph.out

python compute_coefficients.py --dataset_idx 0 --solver 'graph_lasso' --data_filepath 'data/graph' --results_filepath 'results/graph'

python subset_selections.py --selection_method 'top_k' --k 1 --config_option 'bootstrap_graph_lasso' --results_filepath 'results/graph' --data_filepath 'data/graph' --D 1 --J 759

python subset_selections.py --selection_method 'top_k' --k 2 --config_option 'bootstrap_graph_lasso' --results_filepath 'results/graph' --data_filepath 'data/graph' --D 1 --J 759

python subset_selections.py --selection_method 'avg_thresh' --tol 0.5 --config_option 'bootstrap_graph_lasso' --results_filepath 'results/graph' --data_filepath 'data/graph' --D 1 --J 759

python subset_selections.py --selection_method 'inflated_argmax' --epsilon 0.02 --config_option 'bootstrap_graph_lasso' --results_filepath 'results/graph' --data_filepath 'data/graph' --D 1 --J 759

python subset_selections.py --selection_method 'argmax' --config_option 'graph_lasso' --results_filepath 'results/graph' --data_filepath 'data/graph' --D 1 --J 759