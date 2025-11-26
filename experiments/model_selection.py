import numpy as np
from sklearn.linear_model import Lasso
from utils import inflated_argmax
import pysindy as ps
from sklearn.covariance import GraphicalLasso
import pickle
import warnings
import cv2 as cv
import random
import re
from sklearn.tree import DecisionTreeClassifier, export_text

#
def bootstrap(X, y, m):
    
    X = np.array(X); y = np.array(y)

    # randomly sample m different indices
    idxs = np.random.choice(range(X.shape[0]), size = m, replace=False)

    # return X and y only at the selected indices
    return X[idxs, :], y[idxs]

#
def fit_LASSO(X, y, Lambda):
    lasso = Lasso(Lambda) 
    lasso.fit(X,y)
    return lasso

#
def fit_LASSO_bootstrap(X_train, y_train, params):
    '''
    fit_LASSO_bootstrap trains a LASSO model on training data (X_train) to predict labels (y_train), but without one specific 
        instance in the training data (idx_remove)

    X_train (np.array, shape: (# of samples, # of covariates)): training design matrix
    y_train (np.array, shape: (# of samples)): training labels for X_train
    Lambda (float): LASSO hyperparameter
    m (int): number of instances to bootstrap sample for fitting LASSO

    Returns the fitted LASSO model's selected coefficients
    '''

    m = params['m']; Lambda = params['Lambda']

    # compute a bootstrapped dataset
    X_b, y_b = bootstrap(X_train, y_train, m)

    # fit the LASSO model on the reduced training dataset
    b_lasso = fit_LASSO(X_b, y_b, Lambda)

    # return a list of booleans of whether each covariate was selected
    return np.abs(b_lasso.coef_) > 0

def fit_kmeans_bootstrap(X_train, params):
    '''
    fit_kmeans_bootstrap trains kmeans on the training dataset X_train to select the number of clusters 
    '''

    m = params['m']; Lambda = params['Lambda']; max_num_clusters = params['max_num_clusters']

    # compute a bootstrapped dataset
    X_b, _ = bootstrap(X_train, np.empty(X_train.shape), m)

    # fit kmeans on this bootstrapped dataset
    num_clusters = cluster_cv(X_b, max_num_clusters, Lambda)

    # return the selected number of clusters
    return num_clusters

# remove the exact split values of continuous features in the tree
def remove_text_between_delimiters(text, delimiter1, delimiter2):
    # Define the regular expression pattern to match the text between the delimiters
    pattern = f"({re.escape(delimiter1)}.*?{re.escape(delimiter2)})"
    
    # Replace the matched part with just the delimiters
    result = re.sub(pattern, delimiter1 + delimiter2, text)
    
    return result

def fit_decision_tree(X_train, y_train, params):
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Train the model
    dt_classifier.fit(X_train, y_train)
    
    # get the decision tree in text format
    tree_rules = export_text(dt_classifier, feature_names=params['feature_names'])
    return remove_text_between_delimiters(remove_text_between_delimiters(tree_rules, "<=", "\n"), ">" , "\n")

# bootstrapping data and computing the selected decision tree
def fit_decision_tree_bootstrap(X_train, y_train, params):
    m = params['m'] 

    # compute a bootstrapped dataset
    X_b, y_b = bootstrap(X_train, y_train, m)

    # fit a decision tree on this bootstrapped data
    tree = fit_decision_tree(X_b, y_b, params)

    # return the fitted tree
    return np.array([tree]) # returns np.array with one element with the selected tree

def bootstrap_decision_tree(X_train, y_train, params):
    B = params['B']

    # compute B different LASSO models, collect the selected trees
    # shape: (B,1)
    b_decision_trees = np.stack([fit_decision_tree_bootstrap(X_train, y_train, params) for n in range(B)])

    return b_decision_trees

#
def bootstrap_LASSO(X_train, y_train, params):
    '''
    bootstrap_LASSO computes LASSO linear regression models with a bootstrapped training dataset 
    
    X_train (np.array, shape: (# of samples, # of covariates)): training design matrix
    y_train (np.array, shape: (# of samples)): training labels for X_train
    B (int): number of bootstrapped datasets
    m (int): size of boostrapped sample
    Lambda (float): LASSO hyperparameter

    Returns
    '''
    B = params['B']

    # compute B different LASSO models, collect the selected covariates
    # shape: (B, # covariates)
    b_lasso_coeffs = np.stack([fit_LASSO_bootstrap(X_train, y_train, params) for n in range(B)])


    return b_lasso_coeffs 

def bootstrap_kmeans(X_train, params):
    B = params['B']

    # compute B different LASSO models, collect the selected covariates
    # shape: (B, 1)
    b_lasso_coeffs = np.stack([[fit_kmeans_bootstrap(X_train, params)] for n in range(B)])

    return b_lasso_coeffs

#
def get_set_probabilities(coeffs, params):

    '''
    get_set_frequencies computes the frequency of each unique set of covariates selected by LASSO to have nonzero coefficients

    coeffs (shape: (# sets, # of covariates))

    Returns the unique set of selected coefficients 
    '''

    # get unique covariate selections
    unique_coeff_sets = np.unique(coeffs, axis = 0)

    # get counts for each unique coefficient set
    counts = [(coeffs == unique_coeff_sets[i,:]).all(-1).sum() # compute the count for the unique coefficient set
              for i in range(unique_coeff_sets.shape[0])]      # iterate through the # of total coefficient sets

    #convert unique_coeff_sets into a list of tuples
    unique_coeff_sets = [tuple(l) for l in unique_coeff_sets.tolist()]


    # return a dictionary, keys: unique sets of covariates selected, values: counts for each unique set
    w = dict(zip(unique_coeff_sets, counts/ sum(counts)))

    # save the dictionary
    with open(params['set_probabilities_filepath'], 'wb') as f:

        pickle.dump(w, f)
    #print(w)
    return w

#
def top_k_sets(d, w, k):
    top_k_w = np.sort(w)[-k:]
    return [key for key in d.keys() if (d[key] in top_k_w)]

#
def exceptions(params):
    selection_method = params['selection_method']
    epsilon = params["epsilon"]
    k = params["k"]
    solver = params["solver"]
    tspan = params["tspan"]
    poly_order = params['poly_order']
    tol = params['tol']

    if not(selection_method in ["inflated_argmax", "argmax", "top_k", "avg_thresh"]):
        raise Exception('Must input one of the following options for selection_method: "inflated_argmax", "argmax", "top_k", "avg_thresh" ')

    if selection_method == "inflated_argmax" and epsilon == None :

        raise Exception('Must input a floating point number for epsilon if selection_method="inflated_argmax"')
    if selection_method == "top_k" and k == None:
        raise Exception("Must input an integer for k if selection_method = 'top_k'")
    if selection_method == "avg_thresh" and tol == None:
        raise Exception("Must input a float for tol if selection_method = 'avg_thresh'")
    
    if not(solver in ["sindy", "lasso", "graph_lasso", "kmeans", "decision_tree"]):
        raise Exception("solver must be 'sindy', 'lasso', 'graph_lasso', 'kmeans', or 'decision_tree'")
    
    if solver == "sindy": 
        cont = True
        try:
            tspan == None or poly_order == None
            cont = False
        except:
            pass
        if not(cont):
            raise Exception("Must specify tspan and poly_order when using SINDy")
    return

#
def return_selected_subsets(coefs, params):
    '''
    Computes the set of selected high probability subsets based on the selection_method decision criterion

    subset_dict (shape: # bootstraps, # covariates): 2D array containing boolean values denoting whether the covariate was selected for the ensemble member
    selection_method (str): method for returning the selected subset. Options: ["argmax", "inflated_argmax", "top_k"]
    epsilon (float): value for inflating the set that argmax returns. This cannot be None if selection_method="inflated_argmax"
    k (int): integer value for how many top high probability subsets to return. This cannot be None if selection_method="top_k" 
    '''
    selection_method = params['selection_method']
    epsilon = params['epsilon']
    k = params['k']
    tol = params['tol']

    #print("testing")
    exceptions(params)


    if selection_method == "avg_thresh":

        # compute inclusion probabilities
        ip = np.mean(np.abs(np.array(coefs)) > 0, axis = 0)


        # zero out the coefficients whose inclusion probability is lower than tol
        thresh_coef = (ip > tol) 

        # return the set of selected coefficients
        return [tuple(thresh_coef)]
    else:

        # compute probabilities of each unique selected covariate subset in the ensemble
        try:
            # if a filepath is already computed, read it in 
            with open(params['set_probabilities_filepath'], 'rb') as f:
                subset_dict = pickle.load(f)


        except: 
            # if the w's haven't been computed yet, compute them
            subset_dict = get_set_probabilities(coefs, params)


        if selection_method == "argmax":

            return [max(subset_dict, key=subset_dict.get)]
        else:
            # get list of probabilities from the set:probabilility dictionary
            dict_to_list = list(subset_dict.items())
            w = np.array([pair[1] for pair in dict_to_list])

            if selection_method == "inflated_argmax":
                # get the indices of which probabilities were selected by argmax^epsilon
                idxs = list(inflated_argmax(w, epsilon))

                # models selected by argmax^epsilon 
                return [dict_to_list[idx][0] for idx in idxs]
                
            else:

                # get top k indices
                top_k = top_k_sets(subset_dict, w, k)

                return top_k

#   
def remove_one_instance(X, idx_remove):
    # remove instance from training data
    X_rm1 = np.delete(X, idx_remove, axis=0)
    return X_rm1

#
def convert_SINDy_results_to_binary(coefs):
    return [tuple(np.abs(np.array(coefs[i]).flatten()) >0) for i in range(len(coefs))]


#
def compute_coefficients(X,params, tspan = None):
    '''
    compute_coefficients computes fitted coefficients training on X (y is the last column in linear regression & decision tree problems)
    '''
    bootstrap = params['bootstrap']
    Lambda = params["Lambda"]
    B = params['B']
    m = params['m']
    solver = params['solver']
    poly_order = params["poly_order"]

    
    if bootstrap:
        if solver == "lasso":

            ######################
            # Bootstrapped LASSO #
            ######################

            x = X[:,:-1]; y = X[:,-1]

            # bootstrapped LASSO fit, grab the empirical probabilities of each unqiue set of selected covariates
            bootstrap_coefs = bootstrap_LASSO(x, y, params)


            # return subsets selected based on the selection method
            return bootstrap_coefs 
        elif solver == "sindy": 

            #######################
            #    Ensemble SINDy   #
            #######################

            # Ensemble SINDy fit
            ensemble_optimizer = ps.STLSQ(threshold = params['tol_optim'],alpha = Lambda)
            model = ps.SINDy(optimizer=ensemble_optimizer, 
                             feature_library= ps.PolynomialLibrary(degree=poly_order))
            model.fit(X, t=tspan, 
                      ensemble=True, 
                      replace=False, 
                      quiet=True, 
                      n_models = B, 
                      n_subset = m)

            # return the selected coefficients
            return model.coef_list
        elif solver == "kmeans":

            ########################
            #   Bootstrap KMEANS   #
            ########################
             
            bootstrap_coefs = bootstrap_kmeans(X, params)
            return bootstrap_coefs
        
        elif solver == "decision_tree":

            #################################
            #    Bootstrap Decision tree    #
            #################################
            x = X[:,:-1]; y = X[:,-1]

            bootstrap = bootstrap_decision_tree(x, y, params)
            return bootstrap
        
        else:
            #########################
            # Bootstrap Graph LASSO #
            #########################
            bootstrap_g_coefs = bootstrap_graph_LASSO(X, params)
            return bootstrap_g_coefs

    else:
        if solver == "lasso":

            ###############
            #    LASSO    #
            ###############  
            x = X[:,:-1]; y = X[:,-1]

            # fit LASSO on the X, y data
            model = fit_LASSO(x,y,Lambda)

            # determine which coefficients were selected 
            model_coef = np.abs(model.coef_) > 0

            # return the binary list of selected coefficients
            return [tuple(model_coef)]
        
        elif solver == "sindy":

            ###############
            #    SINDy    #
            ###############

            # SINDy fit
            model = ps.SINDy(ps.STLSQ( threshold = params['tol_optim'], alpha = Lambda),
                             feature_library=ps.PolynomialLibrary(degree=poly_order))
            model.fit(X, t=tspan)


            # process the SINDy output, return the list of coefficients
            return [model.coefficients()]
        elif solver == "kmeans":

            ###################
            #      KMEANS     #
            ###################
             
            num_clusters = cluster_cv(X, params['max_num_clusters'], params['Lambda'])
            #print(num_clusters)
            return [ [num_clusters] ]
        elif solver == "decision_tree":

            ##########################
            #      Decision Tree     #
            ##########################
            x = X[:,:-1]; y = X[:,-1]
            tree = fit_decision_tree(x, y, params)

            return [[tree]]
        
        else:
            ####################
            #    Graph LASSO   #
            ####################

            g_lasso = fit_graph_LASSO(X, Lambda)
            return[(np.abs(g_lasso.get_precision()) >0).flatten() ]
        
########################################################################
# GRAPH MODULES

def bootstrap_graph(X, m):
    
    X = np.array(X)
    # randomly sample m different indices
    idxs = np.random.choice(range(X.shape[0]), size = m, replace=False)

    # return X and y only at the selected indices
    return X[idxs, :]
#
def fit_graph_LASSO(X, Lambda):
    g_lasso = GraphicalLasso(alpha= Lambda) 
    g_lasso.fit(X)
    return g_lasso

#
def fit_graph_LASSO_bootstrap(X_train, params):
    '''
    fit_LASSO_bootstrap trains a LASSO model on training data (X_train) to predict labels (y_train), but without one specific 
        instance in the training data (idx_remove)

    X_train (np.array, shape: (# of samples, # of covariates)): training design matrix
    Lambda (float): LASSO hyperparameter
    m (int): number of instances to bootstrap sample for fitting LASSO

    Returns the fitted LASSO model's selected coefficients
    '''

    m = params['m']; Lambda = params['Lambda']

    # compute a bootstrapped dataset
    X_b= bootstrap_graph(X_train, m)

    # fit the LASSO model on the reduced training dataset
    b_lasso = fit_graph_LASSO(X_b, Lambda)

    # return a list of booleans of whether each covariate was selected
    return (np.abs(b_lasso.get_precision()) > 0).flatten()

#
def bootstrap_graph_LASSO(X_train, params):
    '''
    bootstrap_LASSO computes LASSO linear regression models with a bootstrapped training dataset 
    
    X_train (np.array, shape: (# of samples, # of covariates)): training design matrix
    y_train (np.array, shape: (# of samples)): training labels for X_train
    B (int): number of bootstrapped datasets
    m (int): size of boostrapped sample
    Lambda (float): LASSO hyperparameter

    Returns
    '''
    B = params['B']

    # compute n_train different LASSO models, collect the selected covariates
    # shape: (n_train, # covariates)
    b_lasso_coeffs = np.stack([fit_graph_LASSO_bootstrap(X_train, params) for n in range(B)])

    return b_lasso_coeffs 

def graph_subset_selection(coefs, params):
    
    return return_selected_subsets(coefs, params)

#########################################################################
# KMEANS MODULES

# clustering functions 
def compute_elbow_slope(k_list, ssds):
    # compute the slope of the elbow plot from kmeans 
    slopes = [(ssds[i]-ssds[i-1])/(k_list[i] - k_list[i-1]) for i in range(1, len(k_list))]
    return slopes

def choose_num_clusters(k_list, slopes, slope_tol):
    # iterate through the list of slopes
    for i in range(len(slopes)):

        # if the slope is less than some slope tolerance, return the preceding k (# of clusters)
        if np.abs(slopes[i]) < slope_tol:
            return k_list[i]
    # if we never meet the slope tolerance criteria, return the maximum # of slopes tried
    warnings.warn("Convergence criteria not met, increase the maximum number of possible clusters")
    return k_list[len(k_list)-1]

def cluster_cv(Z, max_num_clusters, slope_tol):
    '''
    Z: data
    max_num_clusters: maximum number of clusters
    slope_tol: slope value specified that determines the number of clusters selected
    '''
    ###
    # compute fit for each number of clusters 
    ####
    
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
 
    # Set flags (Just to avoid line break in the code)

    previous_ssd = None
    previous_k = None
    count = 0
    for k in range(1,max_num_clusters+1):
        compactness,labels,centers = cv.kmeans(Z,k,None,criteria,10, cv.KMEANS_RANDOM_CENTERS) #cv.KMEANS_PP_CENTERS)
        if count >=1:
            # compute slope
            slope = np.abs((compactness - previous_ssd)/(k - previous_k))

            # if the slope is less than the tolerance, return the previous k

            if slope < slope_tol:
                return previous_k
        
        # store current k and ssd for the next k
        previous_k = k
        previous_ssd = compactness

        count += 1

    # if we exit the above for loop,  
    warnings.warn("Convergence criteria not met, consider increasing max_num_clusters")
    return max_num_clusters
