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

python3 run_optuna_training_testing.py --model_name GB --lang_model_type {lang_type} --scoring_metric {metric} --feature_type {data_type} --pca_key {pca} --results_folder /tigress/jtdu/optuna_training/sklearn/results/ >> /tigress/jtdu/optuna_training/sklearn/sh/opt_{metric}{data_type}{lang_type}.log

""".format(metric=metric, data_type=data_type, lang_type=lang_type, pca = pca)

def SubmitJob(lang_type, metric, data_type, pca, jobdir = os.getcwd()):
    sh_name = 'opt_' + data_type + metric + pca + '.sh'
    with open(jobdir+'/' + sh_name,'w') as f:
        js = jobstring(lang_type, metric, data_type, pca)
        f.write(js)
        os.system('cd {jobdir}; sbatch {sh_name} &'.format(jobdir=jobdir,sh_name=sh_name))

for l in ["Rostlab_Bert"]: #, "UniRep"]:
    for d in ["mutref", "mut", "abs", "ref"]:
        for m in ["auROC"]:
            for method in ["SVC", "NN", "GB"]:
                SubmitJob(l, m, d, "None")
                time.sleep(1)
                if l == "Rostlab_Bert":
                    for p in ["pca100", "pca250","pca500", "pca1000"]:
                        SubmitJob(l, m, d, p)
                        time.sleep(1)

#python3 run_optuna_sklearn.py --model_name GB --lang_model_type Rostlab_Bert --scoring_metric auPRC --feature_type abs --results_folder /tigress/jtdu/optuna_training/sklearn/results/Rostlab_Bert


root, exclude, ref_paths, mut_paths, pca_mats, start, cols, metas = configure(args)
features, labels, input_df, metadata, feature_columns = load_data(ref_paths, mut_paths, start, cols, exclude, metas, args.feature_type, args.split)

    # define training data folder
    root = "/tigress/jtdu/map_language_models/user_files/"

            "d1": root + "d1_primateNegativesDocmPositives_key_Rostlab_Bert_reference.tsv",
            "d2": root + "d2_primateNegativesDocmPositives_key_Rostlab_Bert_reference.tsv",
ref_paths = {
    "d1": root + "d1_primateNegativesDocmPositives_key_Rostlab_Bert_reference.tsv",
    "d2": root + "d2_primateNegativesDocmPositives_key_Rostlab_Bert_reference.tsv",
    "mcf10A": root + "mmc2_newlabs_key_Rostlab_Bert_reference.tsv",
    "maveDB": root + "mavedb_offset_key_Rostlab_Bert_reference.tsv"
}

## Define paths to Bert Vectors of mutant sequences
mut_paths = {
"d1": root + 'd1_primateNegativesDocmPositives_key_Rostlab_Bert_mutant.tsv',
 "d2": root + 'd2_primateNegativesDocmPositives_key_Rostlab_Bert_mutant.tsv',
"mcf10A": root + "mmc2_newlabs_key_Rostlab_Bert_mutant.tsv",
"maveDB": root + "mavedb_offset_key_Rostlab_Bert_mutant.tsv"
}


python3 run_optuna_training_testing.py --model_name GB --n 2 --lang_model_type Rostlab_Bert --training_path --training_alias --testing_path --testing_alias --results_folder /tigress/jtdu/optuna_training/sklearn/results/test/ >> /tigress/jtdu/optuna_training/optuna_via_sklearn/sh/opt_{metric}{data_type}{lang_type}.log
