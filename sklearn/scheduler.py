import os
import time

def jobstring(lang_type, metric, data_type, pca):
    return """#!/bin/bash
#SBATCH --job-name=opt_{metric}{data_type}{lang_type}{pca}      # create a short name for your job
#SBATCH --error=/tigress/jtdu/optuna_training/sklearn/sh/opt_{metric}{data_type}{lang_type}{pca}.err
#SBATCH --output=/tigress/jtdu/optuna_training/sklearn/sh/opt_{metric}{data_type}{lang_type}{pca}.out
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=120G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=23:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate optuna

python3 run_optuna_sklearn.py --model_name GB --lang_model_type {lang_type} --scoring_metric {metric} --feature_type {data_type} --pca_key {pca} --results_folder /tigress/jtdu/optuna_training/sklearn/results/{lang_type}/ >> /tigress/jtdu/optuna_training/sklearn/sh/opt_{metric}{data_type}{lang_type}.log

""".format(metric=metric, data_type=data_type, lang_type=lang_type, pca=pca)

def SubmitJob(lang_type, metric, data_type, pca="None", jobdir = os.getcwd()):
    sh_name = 'opt_' + data_type + pca + metric + '.sh'
    with open(jobdir+'/' + sh_name,'w') as f:
        js = jobstring(lang_type, metric, data_type, pca)
        f.write(js)
        os.system('cd {jobdir}; sbatch {sh_name} &'.format(jobdir=jobdir,sh_name=sh_name))

for l in ["Rostlab_Bert"]: #, "UniRep"]:
    for d in ["mutref", "mut", "abs", "ref"]:
        for p in ["pca100", "pca250", "pca500", "pca1000"]:
            for m in ["auPRC", "auROC"]:
                SubmitJob(l, m, d, p)
                time.sleep(1)

#python3 run_optuna_sklearn.py --model_name GB --lang_model_type Rostlab_Bert --scoring_metric auPRC --feature_type abs --results_folder /tigress/jtdu/optuna_training/sklearn/results/Rostlab_Bert

