"""Optuna optimization of hyperparameters: argument processing module

Processes arguments for each script in the optuna_training library. Returns an
object of class argparse.

  Typical usage examples:
    from run_optuna_args import *
    args = parse_run_optuna_args()
"""
import argparse

def parse_run_optuna_args():
    parser = argparse.ArgumentParser(description="Optuna optimization of hyperparameters.")
    # Define output folder
    parser.add_argument("--results_folder", type=str, default = "optuna_via_sklearn/results/",
                            help="Write path to folder of results. Must end in '/'. ")

    # Optuna parameters
    parser.add_argument("--model_name", type = str, default = "GB",
                        choices = ["GB", "SVC", "NN"],
                        help="Name of Machine Learning algorithm.")
    parser.add_argument("--scoring_metric", type=str, default= "auROC",
                        choices = ["auPRC", "auROC", "auROC_bygene", "accuracy"],
                        help="Full path to directory with labeled examples. ROC, PR, accuracy.")
    parser.add_argument("--n", type=int, default=200, help="Number of models for oputuna to train.")

    # Data specification parameters
    parser.add_argument("--feature_type", type=str, default= "mut",
                        help="Mapping of aa representation between mutant and reference.")
    parser.add_argument("--lang_model_type", type=str, default = "lang_model_type", #choices = ["UniRep", "Rostlab_Bert", "other"],
                        help="Type of language model underlying features.")
    parser.add_argument("--pca_key", type = str, default = "None", help="PCA matrix specified by key in pca_mats. See config file for further specifications.")
    parser.add_argument("--split", type = str, default = "None", help="Number of folds to split crossvalidation data into.")


    args = parser.parse_args()
    return(args)
