import argparse 
import numpy as np
import yaml
import time
from model_selection import compute_coefficients, remove_one_instance
import os

def save_fitted_coefficients(X, tspan, config_option, J, dataset_idx, results_filepath, feature_names = None):
    # read the yaml file to load parameters
    with open(args.config_filepath, 'r') as file:
        yaml_file = yaml.safe_load(file)
    
    # create a dictionary of parameters from the specified configuration option
    params = yaml_file[config_option]

    params['feature_names'] = feature_names

    # create results folder if it doesn't already exist
    os.makedirs(f"{results_filepath}{params['Lambda']}", exist_ok=True)

    if J == None:
        base_coef = compute_coefficients(X,params, tspan = tspan)
        np.save(f"{results_filepath}{params['Lambda']}/{config_option}_J{J}dataset_{dataset_idx}.npy", base_coef)
    else:

        # leave out data from index J
        X_j = remove_one_instance(X, J)
        if config_option in ["sindy", "ensemble_sindy"]:
            tspan_j = remove_one_instance(tspan, J)
        else: 
            tspan_j = None

        LOO_coef = compute_coefficients(X_j,params, tspan = tspan_j)
        np.save(f"{results_filepath}{params['Lambda']}/{config_option}_J{J}_dataset{dataset_idx}.npy", LOO_coef)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_idx", type = int, default = None)
    parser.add_argument("--config_filepath", type = str, default = "config.yaml")
    parser.add_argument("--solver", type = str, help = "options: 'sindy', 'lasso', 'graph', 'decision_tree', or 'kmeans'")
    parser.add_argument("--data_filepath", type = str, default = "data/clusters")
    parser.add_argument("--results_filepath", type = str)
    parser.add_argument("--J", default = None, type = int)
    args = parser.parse_args()

    # load data
    X = np.load(f"{args.data_filepath}/clusters{args.dataset_idx}.npy").astype(np.float32)
    
    tspan = None
        

    n_samples = X.shape[0]

    start_t = time.time()

    # kmeans
    if args.J == None or args.J == "None":
        print("kmeans j none")
        save_fitted_coefficients(X, tspan, "kmeans", None, args.dataset_idx, args.results_filepath)
        print("bootstrap kmeans j none")
        save_fitted_coefficients(X, tspan, "bootstrap_kmeans", None, args.dataset_idx, args.results_filepath)
    else:
        print("kmeans")
        save_fitted_coefficients(X, tspan, "kmeans", args.J, args.dataset_idx, args.results_filepath)
        print("bootstrap kmeans")
        save_fitted_coefficients(X, tspan, "bootstrap_kmeans", args.J, args.dataset_idx, args.results_filepath)
    