#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=<PARTITION_NAME>
#SBATCH --job-name=decision_tree
#SBATCH --output=temp_job/decision_tree.out

python compute_coefficients.py --dataset_idx 0 --solver 'decision_tree' --data_filepath 'data/decision_tree' --results_filepath 'results/decision_tree'

python subset_selections.py --selection_method 'top_k' --k 1 --config_option 'bootstrap_decision_tree' --results_filepath 'results/decision_tree' --data_filepath 'data/decision_tree' --D 1 --J 50

python subset_selections.py --selection_method 'top_k' --k 2 --config_option 'bootstrap_decision_tree' --results_filepath 'results/decision_tree' --data_filepath 'data/decision_tree' --D 1 --J 50

python subset_selections.py --selection_method 'inflated_argmax' --epsilon 0.03 --config_option 'bootstrap_decision_tree' --results_filepath 'results/decision_tree' --data_filepath 'data/decision_tree' --D 1 --J 50

python subset_selections.py --selection_method 'argmax' --config_option 'decision_tree' --results_filepath 'results/decision_tree' --data_filepath 'data/decision_tree' --D 1 --J 50