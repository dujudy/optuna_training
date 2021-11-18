"""Optuna optimization of hyperparameters.

Implements optuna optimization algorithms using sklearn ML algorithms (currently,
GradientBoostingClassifier.) auROC, auPRC or accuracy between datasets can be
used as the optimization objective. Returns an optuna study class.

  Typical usage examples:
    python3 run_optuna_sklearn.py
    python3 run_optuna_sklearn.py --scoring_metric ROC
"""
import argparse
from joblib import dump, load
import optuna
import pandas as pd
import faiss
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from os.path import exists
from os import system

from generate_prediction_probs import *
from load_data import *
from plot_optuna_results import *

def define_model(model_name, params):
    if model_name =="GB":
        return GradientBoostingClassifier(**params)
    else:
        raise SomeError("Model name not valid.")

def train_model(model_name, parameters, train_feats, train_labs, save = False):
    # define model type
    classifier = define_model(model_name, parameters)
    # train model
    classifier.fit(train_feats, train_labs)
    # save model
    if save != False:
        dump(classifier, save + '_model.joblib')
    return classifier

def score_model(parameters, train_feats, train_labs, test_feats, test_labs, metric, model_name):
    # train_model()
    # define model type
    #classifier = define_model(model_name, parameters)
    # train model
    #classifier.fit(train_feats, train_labs)
    classifier = train_model(model_name, parameters, train_feats, train_labs)
    # generating prediction probs for test set
    y_score = classifier.predict_proba(test_feats)[:,1]
    # generate scoring metric
    if metric == "auROC":
        # calculate ROC AUC
        metric = roc_auc_score(test_labs, y_score)
    elif metric == "auPRC":
        # calculate auprc
        precision, recall, thresholds = precision_recall_curve(test_labs, y_score)
        metric = auc(recall, precision)
    elif metric == "accuracy":
        # calculate mean accuracy
        metric = classifier.score(test_feats, test_labs)
    return(metric)

## Unaltered default params
    #loss='deviance', learning_rate=0.1, subsample=1.0, criterion='friedman_mse', min_weight_fraction_leaf=0.0,
    #min_impurity_split=None, max_leaf_nodes=None, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0
    #min_weight_fraction_leaf=0.0

def objective(trial, train, test, type, feats, input_df, metric,  model_name):
    # define model
    params = {
        'max_depth': trial.suggest_int("max_depth", 1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 75, 300),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        #"min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.25),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 25), # make min larger 1--> 5?
        "random_state": 7}
    # train and evaluate models
    fold_1_auc = score_model(params, feats[type][train], labels[type][train], feats[type][test], labels[type][test], metric,  model_name)
    fold_2_auc = score_model(params, feats[type][test], labels[type][test], feats[type][train], labels[type][train], metric,  model_name)
    return 0.5 * (fold_1_auc + fold_2_auc)


def fill_objective(train, test, type, feats, labs, scoring_metric, model_name):
  def filled_obj(trial):
    return objective(trial, train, test, type, feats, labs, scoring_metric, model_name)
  return filled_obj

def optimize_hyperparams(feature_type = "ref", scoring_metric = "PR", n_trials = 200,  model_name = "GB"):
    specified_objective = fill_objective("d1", "d2", feature_type, features, input_df, scoring_metric,  model_name)
    study = optuna.create_study(direction="maximize")
    study.optimize(specified_objective, n_trials = n_trials)
    return(study)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna optimization of hyperparameters.")
    parser.add_argument("--model_name", type = str, default = "GB", choices = ["GB"],
                        help="Name of Machine Learning algorithm.")
    parser.add_argument("--scoring_metric", type=str, default= "auPRc",
                        choices = ["auPRC", "auROC", "accuracy"],
                        help="Full path to directory with labeled examples. ROC, PR, accuracy.")
    parser.add_argument("--feature_type", type=str, default= "ref",
                        choices = ["ref", "mut", "abs", "mutref"],
                        help="Mapping of aa representation between mutant and reference.")
    parser.add_argument("--n", type=int, default=200, help="Number of models for oputuna to train.")
    parser.add_argument("--results_folder", type=str, default = "results/",
                        help="Write path to folder of results. Must end in '/'")
    parser.add_argument("--lang_model_type", type=str, default = "UniRep", choices = ["UniRep", "Rostlab_Bert"],
                        help="Type of language model underlying features. Default: config paths left as is.")
    parser.add_argument("--pca_key", type = str, default = "None", help="PCA matrix specified by key in pca_mats. See config file for further specifications.")

    args = parser.parse_args()

    # Load data
    if args.lang_model_type == "UniRep":
        from config import *
    elif args.lang_model_type == "Rostlab_Bert":
        from config_RostlabBert import *
    features, labels, input_df, metadata, feature_columns = load_data(ref_paths, mut_paths, start, cols, exclude)

    # Apply PCA if applicable
    if args.pca_key != "None":
        print(args.pca_key)
        pca = faiss.read_VectorTransform(pca_mats[args.pca_key])
        newfeat = args.feature_type + "_" + args.pca_key; features[newfeat] = {}; labels[newfeat] = {}; metadata[newfeat] = {};
        for data_name in features[args.feature_type].keys():
            features[newfeat][data_name] = pca.apply_py(np.ascontiguousarray(features[args.feature_type][data_name].astype('float32')))
            labels[newfeat][data_name] = labels[args.feature_type][data_name]
            metadata[newfeat][data_name] = metadata[args.feature_type][data_name]
            del data_name
    args.feature_type = newfeat

    # Check if optuna-trained model already exists
    run_id = args.results_folder + "/" + "{write_type}" + args.lang_model_type + args.feature_type + "_" + args.model_name + args.scoring_metric
    model_path = run_id.format(write_type="d_") + '_model.joblib'
    if exists(model_path):
        # load model
        print("Loading model at: " + model_path)
        final_classifier = load(model_path)
    else:
        # optimize hyperparameters with optuna
        print("Running optuna optimization.")
        optuna_run = optimize_hyperparams(args.feature_type, args.scoring_metric, args.n, args.model_name)
        plot_optuna_results(optuna_run, run_id.format(write_type="optuna_d1d2_"))

        # train final model using optuna's best hyperparameters
        final_classifier = train_model(
            args.model_name,
            optuna_run.best_trial.params,
            features[args.feature_type]["d"],
            labels[args.feature_type]["d"],
            save = model_path)
#            save = run_id.format(write_type = "d_"))

    # generate prediction probabilities
    for data_name in ref_paths:
        if data_name not in ["d1","d2", "d"]:
            print(data_name); print(model_path);
            generate_prediction_probs(final_classifier,
                                      args.model_name + "_opt" + args.scoring_metric,
                                      features[args.feature_type][data_name],
                                      labels[args.feature_type][data_name],
                                      metadata[args.feature_type][data_name],
                                      run_id.format(write_type=data_name + "_d_")
                                     )

    # generate performance plots
    plot_script = "test_ml_models/plot_analyses/"
    for f in ["plot_mcf10A_optuna_comparisons.R", "plot_mavedb_optuna_comparisons.R"]:
        print(f)
        system(plot_script + f)
