import os
import numpy as np

# names a folder to be added in the current working directory named 'temp_job'
temp_job_folder = 'temp_job'


# creates a folder in the current working directory named 'temp_job'
def mkdir_p(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
mkdir_p(temp_job_folder)

def run_job(temp_job_folder, job_name, d,j):
    job_file = '{}/{}_d{}j{}.sh'.format(temp_job_folder, job_name, d, j)
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --time=01:00:00\n") # max amount of time for the job
        fh.writelines("#SBATCH --partition=general\n")
        fh.writelines("#SBATCH --job-name={}_d{}j{}\n".format(job_name, d,j))
        fh.writelines("#SBATCH --output={}\n".format(temp_job_folder+ "/index{}j{}.out".format(d,j)))
        fh.writelines("python compute_coefficients_kmeans.py ") 
        fh.writelines("--dataset_idx {} ".format(d))
        fh.writelines("--results_filepath 'results/clustering_elbow' ")
        if j is not None:
            fh.writelines(f"--J {j}")
    os.system('sbatch {}'.format(job_file)) 

np.random.seed(0)

datasets = np.arange(0,100,1)
n_samples = range(30)


job_name = "k"
for d in datasets:
    run_job(temp_job_folder, job_name, d, None)
    for n in n_samples:
        run_job(temp_job_folder, job_name, d, n)
