"""Optuna optimization of hyperparameters.

Implements optuna optimization algorithms using sklearn ML algorithms (currently,
GradientBoostingClassifier.) auROC, auPRC or accuracy between datasets can be
used as the optimization objective. Returns an optuna study class.

  Typical usage examples:
    python3 run_optuna_sklearn.py
    python3 run_optuna_sklearn.py --scoring_metric ROC
"""

import optuna
from load_data import *
#from sklearn_fns import *
from plot_optuna_results import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from joblib import dump
import argparse

def define_model(model_name, params):
    if model_name =="GB":
        return GradientBoostingClassifier(**params)
    else:
        raise SomeError("Model name not valid.")

def train_model(model_name, parameters, train_feats, train_lab, save = False):
    # define model type
    classifier = define_model(model_name, parameters)
    # train model
    classifier.fit(train_feats, train_labs)
    # save model
    if save:
        save = dump(clf, 'd_' + model_name + '_model.joblib')
    return classifier

def combine_data(dataset_1, dataset_2, feat_type):
    feats[feat_type]["d"] = np.concatenate(feats[feat_type]["d1"], feats[feat_type]["d2"]), axis=0)
    input_df[feat_type]["d"]["label"] = np.concatenate((input_df[feat_type][dataset_1]["label"], input_df[feat_type][dataset_2]["label"]), axis=0)


def score_model(parameters, train_feats, train_labs, test_feats, test_labs, metric, model_name):
    # train_model()
    # define model type
    #classifier = define_model(model_name, parameters)
    # train model
    #classifier.fit(train_feats, train_labs)
    classifier = train_model(model_name, parameters, train_feats, train_lab)
    # generating prediction probs for test set
    y_score = classifier.predict_proba(test_feats)[:,1]
    # generate scoring metric
    if metric == "ROC":
        # calculate ROC AUC
        metric = roc_auc_score(test_labs, y_score)
    elif metric == "PR":
        # calculate auprc
        precision, recall, thresholds = precision_recall_curve(test_labs, y_score)
        metric = auc(recall, precision)
    else:
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
    fold_1_auc = score_model(params, feats[type][train], input_df[type][train]["label"], feats[type][test], input_df[type][test]["label"], metric,  model_name)
    fold_2_auc = score_model(params, feats[type][test], input_df[type][test]["label"], feats[type][train], input_df[type][train]["label"], metric,  model_name)
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
    parser.add.argument("--model_name", type = str, default = "GB",
                        help="Name of Machine Learning algorithm.")
    parser.add_argument("--scoring_metric", type=str, default= "PR",
                        help="Full path to directory with labeled examples. ROC, PR, accuracy.")
    parser.add_argument("--feature_type", type=str, default= "ref",
                        help="Mapping of aa representation between mutant and reference. ref.")
    parser.add_argument("--n", type=int, default=200, help="Number of models for oputuna to train.")
    parser.add_argument("--plot_suffix", type=str, default= "PRd1d2ref",
                        help="Name of study to annotate plots.")

    args = parser.parse_args()

    optuna_run = optimize_hyperparams(args.feature_type, args.scoring_metric, n_trials = args.n, args.model_name)
    plot_optuna_results(optuna_run, args.plot_suffix)
    # define model type
    combine_data("d1", "d2", args.feature_type)
    final_classifier = train_model(args.model_name, parameters = optuna_run.best_trial.params,
        feats[feat_type]["d"] = np.concatenate(feats[feat_type]["d1"], feats[feat_type]["d2"]), axis=0)
        input_df[feat_type]["d"]["label"], save = True)
