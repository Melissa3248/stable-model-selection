# Experiments

## 1. Regression Experiment

### 1.a. Generating regression data (one dataset for each trial)

```save_regression_data.py``` generates the regression dataset and stores it in the ```data/regression_rho0.99``` folder.

```
python save_regression_data.py 
```

Figure 7 (arXiv version) visualizing the regression covariate covariance matrix is visualized in ```choose_lambda.ipynb```.

### 1.b. Choosing the LASSO penalization $\lambda$

```
choose_lambda.ipynb
```

The above notebook was used to select an appropriate value of $\lambda$ for our generated dataset, and the corresponding chosen value was input into ```config.yaml``` in the Lambda parameter under the "lasso" and "bootstrap_lasso" configuration options.

### 1.c. Submit jobs to train LASSO and subbagged LASSO for each dataset $\mathcal{D}_j$, each leave-one-out (LOO) dataset $\mathcal{D}_j^{\setminus i}$ for $i\in [n]$, and bags $\mathcal{D}^b$ for $b\in[B]$

```submit_parallel_jobs_regression.py``` paralellizes across the dataset dimension to compute and save the coefficients for all fitted models 

```
python submit_parallel_jobs_regression.py
```

### 1.d. Compute the model selections across different selection methods

```subset_selections.py``` computes the selected models based on the input selection method. An example run to compute $\text{argmax}\circ\mathcal{A}$:

```
python subset_selections.py --selection_method "argmax" --config_option "lasso" --J 30 --results_filepath "results/regression_rho0.99" --data_filepath "data/regression_rho0.99"
```

```jobs_across_params/submit_parallel_jobs_reg_params.py``` submits parallel jobs for running subset_selections.py when the selection method is the argmax, $\text{argmax}^\varepsilon$, $\text{ip}_\tau$, or $\text{top-}k$ across different $\varepsilon$, $\tau$, and $k$ values.

```
cd jobs_across_params
python submit_parallel_jobs_reg_params.py
```

### 1.e. Results

The main regression results are visualized in ```results.ipynb```.

## 2. Lotka-Volterra Experiment

### 2.a. Generating LV data (one dataset for each trial)

```save_lv_data.py``` generates the noisy LV dataset and stores it in data/lotka_volterra

```
python save_lv_data.py 
```

Figure 2 and 11 (arXiv version) is visualized in ```plot_lv_trajectories.ipynb```.

### 2.b. Choosing the LASSO penalization $\lambda$

```
choose_lambda.ipynb
```

The above notebook was used to select an appropriate value of $\lambda$ and $\omega$, the SINDy optimization hyperparameters, for our generated dataset. The corresponding chosen values were input into ```config.yaml``` in the Lambda parameter for $\lambda$ and tol_optim for $\omega$ under the "sindy" and "ensemble_sindy" configuration options.


### 2.c. Submit jobs to train SINDy and subbagged SINDy for each dataset $\mathcal{D}_j$, each leave-one-out (LOO) dataset $\mathcal{D}_j^{\setminus i}$ for $i\in [n]$, and bags $\mathcal{D}^b$ for $b\in[B]$

```submit_parallel_jobs_sindy.py``` paralellizes across the dataset dimension to compute and save the coefficients for all fitted models.

```
python submit_parallel_jobs_sindy.py
```

### 2.d. Compute the model selections across different selection methods

```subset_selections.py``` computes the selected models based on the input selection method. An example run to compute $\text{argmax}\circ\mathcal{A}$:

```
python subset_selections.py --selection_method "argmax" --config_option "sindy" --J 21 --results_filepath "results/lotka_volterra" --data_filepath "data/lotka_volterra"
```

```jobs_across_params/submit_parallel_jobs_lv_params.py``` submits parallel jobs for running subset_selections.py when the selection method is the argmax, $\text{argmax}^\varepsilon$, $\text{ip}_\tau$, or $\text{top-}k$ across different $\varepsilon$, $\tau$, and $k$ values.

```
cd jobs_across_params
python submit_parallel_jobs_lv_params.py
```
	
### 2.e. Results

The main regression results (Figures 8, 9, and 10, arXiv version) are visualized in ```results.ipynb```.

## 3. Graph Experiment

### 3.a. Downloading the flow cytometry data

Download the flow cytometry data labeled ```13. cd3cd28icam2+u0126.xls``` from  Sachs et al. 2005 and move it to a folder called ```data/graph```: https://www.science.org/doi/abs/10.1126/science.1105809

### 3.b. Choosing the LASSO penalization $\lambda$

```
graphical_lasso.ipynb
```

The above notebook was used to select an appropriate value of $\lambda$ for the graphical LASSO. The corresponding chosen value was input into ```config.yaml``` in the Lambda parameter for $\lambda$ under the "graph_lasso" and "bootstrap_graph_lasso" configuration options.

### 3.c. Submitting jobs to training the graphical LASSO and subbagged graphical LASSO for each LOO dataset $D_j^{\setminus i}$ for $i \in [n]$, and bags $D^b$ for $b \in [B]$.

```
sbatch graph.sh
```

Running the ```graph.sh``` file submits a job to compute the selected models across bootstrap samples and LOO trials and then computes the corresponding stabilities and LOO set sizes reported in Table 2.

### 3.d. Results

The main results of the stabilities and LOO set sizes across different model selection procedures in Table 2 are saved in the ```results/graph``` folder. Code to reproduce Figure 5 (arXiv version) is included in the ```graphical_lasso.ipynb``` notebook.

## 4. Clustering Experiment

### 4.a. Data generation

```save_kmeans_data.py``` generates independent clustering datasets and stores the data in  the folder ```data/clusters```.

```
python save_kmeans_data.py
```

```clustering.ipynb``` visualizes Figure 15 (arXiv version), which is an example clustering dataset for one simulation.

### 4.b. Submitting jobs to select the number of clusters for each dataset $\mathcal{D}_j$, each leave-one-out (LOO) dataset $\mathcal{D}_j^{\setminus i}$ for $i\in [n]$, and bags $\mathcal{D}^b$ for $b\in[B]$


```submit_parallel_jobs_kmeans.py``` paralellizes across the dataset dimension to compute and save the selected number of clusteres for all fitted models.

```
python submit_parallel_jobs_kmeans.py
```

### 4.c. Compute the model selections across different selection methods

```clustering.sh``` contains example commmand lines to run each of the model selection methods for clustering.

```
sbatch clustering.sh
```

### 4.d. Results

The main results of the stabilities and LOO set sizes across different model selection procedures are saved in the ```results/clustering_elbow``` folder. Code to reproduce Figure 6 (arXiv version) is included in the ```clustering.ipynb``` notebook.


## 5. Decision Tree Experiment

### 5.a. Downloading the single cell transcriptomics data

Download the single cell transcriptomics data labeled ```murine_expression.csv``` and ```murine_annotations.csv``` from  Veleslavov and Stumpf (2020) and move it to a folder called ```data/decision_tree```: https://zenodo.org/records/4342011  

### 5.b. Submitting jobs to training the decision trees and subbagged decision trees for each LOO dataset $D_j^{\setminus i}$ for $i \in [n]$, and bags $D^b$ for $b \in [B]$.

```
sbatch decision_tree.sh
```

Running the ```decision_tree.sh``` file submits a job to compute the selected models across bootstrap samples and LOO trials and then computes the corresponding stabilities and LOO set sizes.

### 5.d. Results

The main results of the stabilities and LOO set sizes across different model selection procedures are saved in the ```results/decision_tree``` folder. Code to visualize the data and results are shown in ```decision_tree.ipynb.```


## 6. Supplementary Material

Table 3 and Figure 10 in the supplementary material are created from running the LV section of ```choose_lambda.ipynb```. 


To run jobs across different numbers of bags, run 

```
cd jobs_across_params
python jobs_across_bags_reg.py
```

for the regression example, and 

```
cd jobs_across_params
python jobs_across_bags_lv.py
```

for the Lotka-Volterra example.


Figures in the supplementary material are visualized in ```appendix_vis.ipynb```.



