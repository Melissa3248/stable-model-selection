import numpy as np
import os

# number of covariates
d = 20

# number of important covariates 
important= 2 

# number of samples 
n= 30

# covariate correlation structure
rho = 0.99
C = np.eye(d)
C[0,1] = rho; C[1,0] = rho
C[2,3] = rho; C[3,2] = rho
C[3,4] = rho; C[4,3] = rho
C[2,4] = rho; C[4,2] = rho



# create 200 new datasets contaminated with noise
for i in range(200):
    X = np.random.multivariate_normal(np.repeat([0], d), C, size = n)
    X_important  = X[:,[0,2]]

    # construct y with some noise
    beta = [1,1]
    y = X_important @ beta + np.random.normal(0,0.3,size = n) #0.5,size = n)

    # combine X_train and y_train data into one matrix
    data = np.concatenate((X,np.array([y]).reshape(X.shape[0],1)), axis = 1)

    # save the data
    os.makedirs(f"data/regression_rho{rho}", exist_ok = True)
    if i <200:
        np.save(f"data/regression_rho{rho}/reg_{i}.npy", data)
    else: 
        np.save(f"data/regression_rho{rho}/reg_CV.npy", data)
    