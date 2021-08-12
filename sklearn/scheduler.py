import os
import time

def jobstring(metric, data_type):
    return """#!/bin/bash
#SBATCH --job-name=opt_{metric}{data_type}       # create a short name for your job
#SBATCH --error=/tigress/jtdu/optuna_training/sklearn/sh/opt_{metric}{data_type}.err
#SBATCH --output=/tigress/jtdu/optuna_training/sklearn/sh/opt_{metric}{data_type}.out
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=120G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=23:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate optuna

python3 ../run_optuna_sklearn.py --model_name GB --scoring_metric {metric} --feature_type {data_type} --results_folder /tigress/jtdu/optuna_training/sklearn/results/ >> opt_{metric}{data_type}.log

""".format(metric=metric, data_type)

def SubmitJob(metric, data_type,jobdir = os.getcwd()):
    sh_name = 'opt_' + data_type + metric + '.sh'
    with open(jobdir+'/' + sh_name,'w') as f:
        js = jobstring(metric, data_type)
        f.write(js)
        os.system('cd {jobdir}; sbatch {sh_name}.sh &'.format(jobdir=jobdir,sh_name=sh_name))

for d in ["mutref", "mut", "abs"]:
    for m in ["auPRC", "auROC"]:
        SubmitJob(m, d)
        time.sleep(1)
