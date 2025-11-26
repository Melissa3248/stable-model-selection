import numpy as np
import os 

np.random.seed(9)

if __name__ == "__main__":
    D =100 # number of datasets
    m = 80 # number of samples per generated dataset

    for i in range(D):

        rng = np.random.RandomState()
        
        n1 = 5
        n2 = 5
        n3 = 20
    
        # Two large, dense clusters
        cluster1 = rng.normal(loc=1.5, scale=1, size=(n1, 2))
        cluster2 = rng.normal(loc=3.5, scale=1, size=(n2, 2))
        cluster3 = rng.normal(loc=2.5, scale=0.3, size=(n3, 2))
        
    
        X = np.vstack([cluster1, cluster2, cluster3])

        # save the data 
        os.makedirs("data/clusters", exist_ok = True)
        np.save(f"data/clusters/clusters{i}.npy", X)

        np.save(f"data/clusters/classes{i}.npy", np.concatenate( (np.repeat([0],n1), np.repeat([1],n2), np.repeat([2],n3) ) ).astype(str).astype(int))