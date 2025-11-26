import os

# names a folder to be added in the current working directory named 'temp_job'
temp_job_folder = 'temp_job'


# creates a folder in the current working directory named 'temp_job'
def mkdir_p(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
mkdir_p(temp_job_folder)

def run_job(temp_job_folder, job_name, n):
    job_file = '{}/{}_{}.sh'.format(temp_job_folder, job_name, n)
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -G1\n") # number of nodes to allocate for this job
        fh.writelines("#SBATCH --time=06:00:00\n") # max amount of time for the job
        fh.writelines("#SBATCH --partition=general\n") #<PARTITION_NAME>\n")
        fh.writelines("#SBATCH --job-name={}_{}\n".format(job_name, n))
        fh.writelines("#SBATCH --output={}\n".format(temp_job_folder+ "/index{}.out".format(n)))
        fh.writelines("python compute_coefficients.py ") 
        fh.writelines("--dataset_idx {} ".format(n))
        fh.writelines("--solver 'sindy' ")
        fh.writelines("--data_filepath 'data/lotka_volterra' ")
        fh.writelines("--results_filepath 'results/lotka_volterra' ")
        fh.writelines("--J 21")
    os.system('sbatch {}'.format(job_file)) 

ts = range(100)

job_name = "lv"
for t_ix in ts:

    run_job(temp_job_folder, job_name, t_ix)
