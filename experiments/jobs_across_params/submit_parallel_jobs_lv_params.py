import os

# names a folder to be added in the current working directory named 'temp_job'
temp_job_folder = 'temp_job'


# creates a folder in the current working directory named 'temp_job'
def mkdir_p(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
mkdir_p(temp_job_folder)

# given input parameters, writes a .sh file and submits a sbatch job
def run_job(temp_job_folder, job_name, param, job_type, num_bags):
    job_file = '{}/{}_{}.sh'.format(temp_job_folder, job_name, param, num_bags)
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -G1")
        fh.writelines("#SBATCH --time=06:00:00\n") # max amount of time for the job
        fh.writelines("#SBATCH --partition=<PARTITION_NAME>\n")
        fh.writelines("#SBATCH --job-name={}_{}\n".format(job_name, param))
        fh.writelines("#SBATCH --output={}\n".format(temp_job_folder+ "/param{}.out".format(param)))

        if job_type == "inflated_argmax":
            fh.writelines("python subset_selections.py --selection_method 'inflated_argmax' ")
            fh.writelines("--epsilon {} ".format(param))
        elif job_type == "top_k":
            fh.writelines("python subset_selections.py --selection_method 'top_k' ") 
            fh.writelines("--k {} ".format(param))
        elif job_type == "avg_thresh":
            fh.writelines("python subset_selections.py --selection_method 'avg_thresh' ")
            fh.writelines("--tol {} ".format(param))
        else:
            import sys; sys.exit("not supported")

        fh.writelines("--config_option 'ensemble_sindy' ")
        fh.writelines("--J 21 --D 100 --B {} ".format(num_bags))
    os.chdir(os.path.dirname(os.getcwd()))
    os.system('sbatch jobs_across_params/{}'.format(job_file)) 
    os.chdir("jobs_across_params")

b = 7000

# run base algorithm
os.chdir(os.path.dirname(os.getcwd()))
os.system(f"python subset_selections.py --selection_method 'argmax' --config_option 'sindy' --J 21 --D 100 --B {b} --data_filepath 'data/lotka_volterra' --results_filepath 'results/lotka_volterra'")
os.chdir("jobs_across_params")

# average threshold
ts = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.63,0.7,0.8,0.9,0.99]
job_name = "tol"

for t_ix in ts:

    run_job(temp_job_folder, job_name, t_ix, "avg_thresh", b)


#epsilon
ts = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]
job_name = "reps"

for t_ix in ts:

    run_job(temp_job_folder, job_name, t_ix, "inflated_argmax", b)

# top k
ts = [1,2,3,4,5,6,7,8,9,10,11,12]
job_name = 'k'

for t_ix in ts:

    run_job(temp_job_folder, job_name, t_ix, "top_k",b)
