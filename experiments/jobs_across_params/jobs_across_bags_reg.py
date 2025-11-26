import os

# run this code in the above working directory
os.chdir(os.path.dirname(os.getcwd()))


num_bags = [10,100,1000,5000]
epsilon = 0.8
tau = 0.3
k = 2

#epsilon
for b in num_bags:
    # epsilon 
    os.system(f"python subset_selections.py --selection_method 'inflated_argmax' --epsilon {epsilon} --config_option 'bootstrap_lasso' --J 30 --D 200 --B {b} --data_filepath 'data/regression_rho0.99' --results_filepath 'results/regression_rho0.99' ")

    # tau 
    os.system(f"python subset_selections.py --selection_method 'avg_threshold' --tol {tau} --config_option 'bootstrap_lasso' --J 30 --D 200 --B {b} --data_filepath 'data/regression_rho0.99' --results_filepath 'results/regression_rho0.99' ")

    # k
    os.system(f"python subset_selections.py --selection_method 'top_k' --k {k} --config_option 'bootstrap_lasso' --J 30 --D 200 --B {b} --data_filepath 'data/regression_rho0.99' --results_filepath 'results/regression_rho0.99' ")

    # argmax
    os.system(f"python subset_selections.py --selection_method 'top_k' --k 1 --config_option 'bootstrap_lasso' --J 30 --D 200 --B {b} --data_filepath 'data/regression_rho0.99' --results_filepath 'results/regression_rho0.99' ")

