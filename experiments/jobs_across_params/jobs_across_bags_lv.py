import os

# run this code in the above working directory
os.chdir(os.path.dirname(os.getcwd()))


num_bags = [10,100,1000,5000]
epsilon = 0.09
tau = 0.63
k = 2

#epsilon
for b in num_bags:
    # epsilon 
    os.system(f"python subset_selections.py --selection_method 'inflated_argmax' --epsilon {epsilon} --config_option 'ensemble_sindy' --J 21 --D 100 --B {b} --data_filepath 'data/lotka_volterra' --results_filepath 'results/lotka_volterra' ")

    # tau 
    os.system(f"python subset_selections.py --selection_method 'avg_threshold' --tol {tau} --config_option 'ensemble_sindy' --J 21 --D 100 --B {b} --data_filepath 'data/lotka_volterra' --results_filepath 'results/lotka_volterra' ")

    # k
    os.system(f"python subset_selections.py --selection_method 'top_k' --k {k} --config_option 'ensemble_sindy' --J 21 --D 100 --B {b} --data_filepath 'data/lotka_volterra' --results_filepath 'results/lotka_volterra' ")

    # argmax
    os.system(f"python subset_selections.py --selection_method 'top_k' --k 1 --config_option 'ensemble_sindy' --J 21 --D 100 --B {b} --data_filepath 'data/lotka_volterra' --results_filepath 'results/lotka_volterra' ")

