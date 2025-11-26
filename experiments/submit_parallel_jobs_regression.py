import os

# names a folder to be added in the current working directory named 'temp_job'
temp_job_folder = 'temp_job'


# creates a folder in the current working directory named 'temp_job'
def mkdir_p(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
mkdir_p(temp_job_folder)

# given input parameters, writes a .sh file and submits a sbatch job
def run_job(temp_job_folder, job_name, n):
    job_file = '{}/{}_{}.sh'.format(temp_job_folder, job_name, n)
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -G1\n") # number of nodes to allocate for this job
        fh.writelines("#SBATCH --time=08:00:00\n") # max amount of time for the job
        fh.writelines("#SBATCH --partition=<PARTITION_NAME>\n")
        fh.writelines("#SBATCH --job-name={}_{}\n".format(job_name, n))
        fh.writelines("#SBATCH --output={}\n".format(temp_job_folder+ "/n{}.out".format(n)))
        fh.writelines("python compute_coefficients.py ") # python script
        fh.writelines("--dataset_idx {} ".format(n))
        fh.writelines("--solver 'lasso' ")
        fh.writelines("--data_filepath 'data/regression_rho0.99' ")
        fh.writelines("--results_filepath 'results/regression_rho0.99' ")
        fh.writelines("--J 30")
    os.system('sbatch {}'.format(job_file)) 

ts = [0]#range(200)

job_name = "reg"
for t_ix in ts:

    run_job(temp_job_folder, job_name, t_ix)