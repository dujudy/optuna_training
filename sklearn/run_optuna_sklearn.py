"""Optuna optimization of hyperparameters.

Implements optuna optimization algorithms using sklearn ML algorithms (currently,
GradientBoostingClassifier.) auROC, auPRC or accuracy between datasets can be
used as the optimization objective. Returns an optuna study class.

  Typical usage examples:
    python3 run_optuna_sklearn.py
    python3 run_optuna_sklearn.py --scoring_metric ROC
"""

import argparse
from joblib import dump
import ipdb
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
#from sklearn_fns import *
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
        dump(classifier, save + "_" + model_name + '_model.joblib')
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
    if metric == "ROC":
        # calculate ROC AUC
        metric = roc_auc_score(test_labs, y_score)
    elif metric == "PR":
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
    parser.add_argument("--scoring_metric", type=str, default= "PR",
                        choices = ["PR", "ROC", "accuracy"],
                        help="Full path to directory with labeled examples. ROC, PR, accuracy.")
    parser.add_argument("--feature_type", type=str, default= "ref",
                        choices = ["ref", "mut"],
                        help="Mapping of aa representation between mutant and reference. ref.")
    parser.add_argument("--n", type=int, default=200, help="Number of models for oputuna to train.")
    parser.add_argument("--plotname_prefix", type=str, default= "test",
                        help="Prefix for filename: Optuna performance plots.")
    parser.add_argument("--modelname_prefix", type=str, default= "test",
                        help="Prefix for filename: hypertuned ML model.")
    parser.add_argument("--testing", action  ='store_true',
                        help="Boolean. If true, run ipdb.")
    args = parser.parse_args()

    # optimize hyperparameters with optuna
    optuna_run = optimize_hyperparams(args.feature_type, args.scoring_metric, args.n, args.model_name)
    plot_optuna_results(optuna_run, args.plotname_prefix)

    # train final model using optuna's best hyperparameters
    final_classifier = train_model(
        args.model_name,
        optuna_run.best_trial.params,
        features[args.feature_type]["d"],
        labels[args.feature_type]["d"],
        save = args.modelname_prefix)

    # optional testing
    if args.testing:
        ipdb.set_trace(context = 7)

#clf = load(args.modelname_prefix+ "_" + args.model_name + '_model.joblib')