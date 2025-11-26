import argparse 
import numpy as np
import yaml
import time
import os
import pandas as pd
from scipy.io import arff
from model_selection import compute_coefficients, remove_one_instance

def save_fitted_coefficients(X, tspan, config_option, J, dataset_idx, results_filepath, feature_names = None):
    # read the yaml file to load parameters
    with open(args.config_filepath, 'r') as file:
        yaml_file = yaml.safe_load(file)
    
    # create a dictionary of parameters from the specified configuration option
    params = yaml_file[config_option]

    params['feature_names'] = feature_names

    if J == None:
        base_coef = compute_coefficients(X,params, tspan = tspan)
        np.save(f"{results_filepath}/{config_option}_J{J}dataset_{dataset_idx}.npy", base_coef)
    else:

        # leave out data from index J
        X_j = remove_one_instance(X, J)
        if config_option in ["sindy", "ensemble_sindy"]:
            tspan_j = remove_one_instance(tspan, J)
        else: 
            tspan_j = None

        LOO_coef = compute_coefficients(X_j,params, tspan = tspan_j)
        np.save(f"{results_filepath}/{config_option}_J{J}_dataset{dataset_idx}.npy", LOO_coef)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_idx", type = int, default = None)
    parser.add_argument("--config_filepath", type = str, default = "config.yaml")
    parser.add_argument("--solver", type = str, help = "options: 'sindy', 'lasso', 'graph', 'decision_tree', or 'kmeans'")
    parser.add_argument("--data_filepath", type = str)
    parser.add_argument("--results_filepath", type = str)
    parser.add_argument("--J", default = None, type = int)
    args = parser.parse_args()

    # create the results path if it does not already exit
    if not os.path.exists(args.results_filepath):
        os.makedirs(args.results_filepath)

    # load X and tspan
    if args.solver =="sindy":
        X = np.load(f"{args.data_filepath}/lv_{args.dataset_idx}.npy")

        tspan = np.load(f"{args.data_filepath}/tspan.npy")
    elif args.solver == "lasso": 
        X = np.load(f"{args.data_filepath}/reg_{args.dataset_idx}.npy")
        tspan = None
    elif args.solver == "kmeans":
        tspan = None

        #'''
        arff_file = arff.loadarff('clustering-benchmark/src/main/resources/datasets/artificial/elly-2d10c13s.arff')
        df = pd.DataFrame(arff_file[0])
        Z =df['x']; Y = df['y']
        X = np.stack((Z,Y),axis = 1).astype(np.float32)
        #'''
    elif args.solver == "decision_tree":
        data = pd.read_csv("data/murine_expression_reduced.csv")

        Z = data.drop('Class', axis = 1)
        y = np.array(data['Class'])

        # get feature names 
        feature_names = Z.columns

        print(np.unique(y))

        print(np.array(Z).shape,np.expand_dims(y, axis=1).shape)

        X = np.hstack((np.array(Z),np.expand_dims(y, axis=1)))
        
        tspan = None
        

    else:
        X = pd.read_excel("data/graph/13. cd3cd28icam2+u0126.xls") 
        tspan = None

    # if no J is input, iterate to remove datapoints one at a time for the entire dataset
    if args.J ==None:
        args.J = X.shape[0]

    start_t = time.process_time()

    # don't leave out any variables
    if args.solver == "sindy":
        # base sindy
        save_fitted_coefficients(X, tspan, "sindy", None, args.dataset_idx, args.results_filepath)

        # ensemble sindy
        save_fitted_coefficients(X, tspan, "ensemble_sindy", None, args.dataset_idx, args.results_filepath)
    elif args.solver == "lasso":
        # base lasso
        save_fitted_coefficients(X, tspan, "lasso", None, args.dataset_idx, args.results_filepath)

        # ensemble lasso
        save_fitted_coefficients(X, tspan, "bootstrap_lasso", None, args.dataset_idx, args.results_filepath)
    elif args.solver == "kmeans":
        # base kmeans
        save_fitted_coefficients(X, tspan, "kmeans", None, args.dataset_idx, args.results_filepath)

        # ensemble kmeans
        save_fitted_coefficients(X, tspan, "bootstrap_kmeans", None, args.dataset_idx, args.results_filepath)
    elif args.solver == "decision_tree":
        # base decision tree classifier
        save_fitted_coefficients(X, tspan, "decision_tree", None, args.dataset_idx, args.results_filepath)

        # ensemble decision tree
        save_fitted_coefficients(X, tspan, "bootstrap_decision_tree", None, args.dataset_idx, args.results_filepath)
    else:
        # base graph lasso
        save_fitted_coefficients(X, tspan, "graph_lasso", None, args.dataset_idx, args.results_filepath)

        # ensemble graph lasso
        save_fitted_coefficients(X, tspan, "bootstrap_graph_lasso", None, args.dataset_idx, args.results_filepath)

    
    # leave out one J at a time
    for J in range(args.J):
        if J % 10 == 0:
            print(J)

        if args.solver == "sindy":
            # base sindy
            save_fitted_coefficients(X, tspan, "sindy", J, args.dataset_idx, args.results_filepath)

            # ensemble sindy
            save_fitted_coefficients(X, tspan, "ensemble_sindy", J, args.dataset_idx, args.results_filepath)
        elif args.solver == "lasso":
            # base lasso
            save_fitted_coefficients(X, tspan, "lasso", J, args.dataset_idx, args.results_filepath)

            # ensemble lasso
            save_fitted_coefficients(X, tspan, "bootstrap_lasso", J, args.dataset_idx, args.results_filepath)
        elif args.solver == "kmeans":
            # base kmeans
            save_fitted_coefficients(X, tspan, "kmeans", J, args.dataset_idx, args.results_filepath)

            # ensemble kmeans
            save_fitted_coefficients(X, tspan, "bootstrap_kmeans", J, args.dataset_idx, args.results_filepath)
        elif args.solver == "decision_tree":
            # base decision tree classifier
            save_fitted_coefficients(X, tspan, "decision_tree", J, args.dataset_idx, args.results_filepath)

            # ensemble decision tree
            save_fitted_coefficients(X, tspan, "bootstrap_decision_tree", J, args.dataset_idx, args.results_filepath)
        else:
            # base graph lasso
            save_fitted_coefficients(X, tspan, "graph_lasso", J, args.dataset_idx, args.results_filepath)

            # ensemble graph lasso
            save_fitted_coefficients(X, tspan, "bootstrap_graph_lasso", J, args.dataset_idx, args.results_filepath)
    print(time.process_time() - start_t)
    
    
    
    