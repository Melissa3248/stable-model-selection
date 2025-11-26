from model_selection import convert_SINDy_results_to_binary, return_selected_subsets
import argparse
import yaml
import numpy as np
import os
import re 

def subset_selection(coefs, params):
    bootstrap = params['bootstrap']
    solver = params['solver']


    if bootstrap:
        if solver == "lasso":
            #######################
            #    Ensemble LASSO   #
            #######################

            # return subsets selected based on the selection method
            return return_selected_subsets(coefs, params)
        elif solver == "sindy": 

            #######################
            #    Ensemble SINDy   #
            #######################
            
            # preprocess coefficient values
            ensemble_coefs = convert_SINDy_results_to_binary(coefs)

            # return the selected coefficients
            return return_selected_subsets(ensemble_coefs, params)

        elif solver == "kmeans":
            #######################
            #   Ensemble kmeans   #
            #######################
            return return_selected_subsets(coefs, params)
        
        else: 
            ############################
            #    Ensemble Graph LASSO  #
            ############################
            return return_selected_subsets(coefs,params)
    else:
        if solver == "lasso":

            ###############
            #    LASSO    #
            ###############  

            # return the binary list of selected coefficients
            return [tuple(coefs.squeeze())]
        elif solver == "sindy":

            ###############
            #    SINDy    #
            ###############
            # process the SINDy output, return the binary list of selected coefficients
            return convert_SINDy_results_to_binary(coefs)
        elif solver == 'kmeans':
            ###############
            return coefs
        elif solver == "decision_tree":
            return coefs
        else:
            ###############
            # Graph LASSO #
            ###############
            return [tuple(coefs.squeeze())]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_method", help='options: "argmax", "inflated_argmax", "top_k"', type=str)
    parser.add_argument("--config_option", type=str)
    parser.add_argument("--config_filepath", type=str, default = "config.yaml")
    parser.add_argument("--epsilon", type=float, default = None)
    parser.add_argument("--tol", type = float, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--results_filepath", type = str, default = "results/lotka_volterra") 
    parser.add_argument("--data_filepath", type = str, default = "data/lotka_volterra") 
    parser.add_argument("--D", type = int, default = 100)
    parser.add_argument("--J", type = int)
    parser.add_argument("--B", default = None, type = int)
    args = parser.parse_args()

    # read the yaml file to load parameters
    with open(args.config_filepath, 'r') as file:
        yaml_file = yaml.safe_load(file)
    
    # create a dictionary of parameters from the specified configuration option
    params = yaml_file[args.config_option]

    # add to the params dictionary the parsed arguments
    params['selection_method'] = args.selection_method
    params['epsilon'] = args.epsilon
    params['k'] = args.k
    params['tol'] = args.tol

    if args.B != None:
        params['B'] = args.B

    if args.config_option in ["sindy", "ensemble_sindy"]:
        params['tspan'] = np.load(f"{args.data_filepath}/tspan.npy")
        params['truth'] = (False, True, False, False, True, False, 
                        False, False, True, False, True, False)
    if args.config_option in ['kmeans', 'bootstrap_kmeans']:
        params['truth'] = (3,)
    else: 
        params['tspan'] = None
        params['truth'] = tuple(np.concatenate((np.array([True,False,True, False, False]),np.repeat(False, 15)))) 



    np.random.seed(0)
    files = os.listdir(args.results_filepath + f"{params['Lambda']}")
    J = np.unique([int(re.findall(r'J(\d+)', file)[0]) for file in files if not("None" in file)])

    D = range(args.D)

    results = []

    for d in range(args.D): # iterate through the datasets
        print(d)
        if params['B'] is None or params['B'] == "None":
            full_coef = np.expand_dims(np.load(f"{args.results_filepath}{params['Lambda']}/{args.config_option}_JNonedataset_{d}.npy"),1)
        else:

            full_coef = np.load(f"{args.results_filepath}{params['Lambda']}/{args.config_option}_JNonedataset_{d}.npy")[:params['B'],:]
        

        params['set_probabilities_filepath'] = args.results_filepath + f"{params['Lambda']}"+  '/prob_weights{}_dataset{}JNone.pkl'.format(params['B'], d)

        if args.config_option in ["sindy", "ensemble_sindy"]:
            full_select = subset_selection(convert_SINDy_results_to_binary(full_coef), params)
        else: 
            
            full_select = subset_selection(full_coef,params)

        # get number of samples in the data 
        n_samples = len(J)

        # keep a count of whether the selected set(s) using the full dataset vs -1 dataset have any overlap 
        count_set_overlap = 0

        # measure how large the returned set is when trained on the full dataset
        num_returned_sets = len(full_select)

        # count how many covariates were selected on average for returned sets
        avg_covariates_selected = np.mean([sum(full_select[i]) for i in range(len(full_select))])

        # keep track of whether the true set is in the returned selection(s)
        count_truth_overlap = 0
        
        truth_in_full_coef = params['truth'] in full_select
        LOO_w_acc = []


        for j in J: # iterate through the LOO results

            if params['B'] is None:
                rm1_coef = np.expand_dims(np.load(f"{args.results_filepath}{params['Lambda']}/{args.config_option}_J{j}_dataset{d}.npy"),1)
            else:
                rm1_coef = np.load(f"{args.results_filepath}{params['Lambda']}/{args.config_option}_J{j}_dataset{d}.npy")#[:params['B'],:]
            
            # create a new weights based on the dataset and which training instance was left out
            params['set_probabilities_filepath'] = args.results_filepath + f"{params['Lambda']}"+ '/prob_weights{}_dataset{}J{}.pkl'.format(params['B'], d,j)
            if args.config_option in ['sindy', 'ensemble_sindy']:
                rm1_select = subset_selection(convert_SINDy_results_to_binary(rm1_coef), params)
            else:
                rm1_select = subset_selection(rm1_coef, params)
            truth_in_LOO_select = (params['truth'] in rm1_select)*1
            LOO_w_acc.append(truth_in_LOO_select/len(rm1_select))

            #count_selected_covariates += np.mean([sum(rm1_select[i]) for i in range(len(rm1_select))])

            # check if the selected subset(s) on the full training data has any overlap with the selected subset(s) on the training data -1 point
            if sum([full_select[i] == rm1_select[j] for i in range(len(full_select)) for j in range(len(rm1_select))]) !=0: 

                # if there's overlap, add it to the overall count
                count_set_overlap += 1

            
            # check if the selected subset(s) contain the truth
            if params['truth'] in rm1_select:
                # if the truth is contained in the selected subset(s), keep track of it
                count_truth_overlap += 1
            
        results.append([count_set_overlap/n_samples, num_returned_sets, avg_covariates_selected , truth_in_full_coef, count_truth_overlap/n_samples, np.mean(LOO_w_acc)])
    print(np.mean(np.array(results)[:,1]))
    print(np.array(results))
    np.save(f"{args.results_filepath}{params['Lambda']}/{args.config_option}_selection_method_{params['selection_method']}_epsilon{params['epsilon']}_k{params['k']}_tol{params['tol']}B{params['B']}.npy",np.array(results))
