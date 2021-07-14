"""
Optuna optimization of hyperparameters
"""
import optuna
from load_data import *
from sklearn_fns import *
from plot_optuna_results import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def score_model(parameters, train_feats, train_labs, test_feats, test_labs, metric):
    # define model type
    classifier = GradientBoostingClassifier(**parameters)
    # train model
    classifier.fit(train_feats, train_labs)
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

def objective(trial, train, test, type, feats, input_df, metric):
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
    fold_1_auc = score_model(params, feats[type][train], input_df[type][train]["label"], feats[type][test], input_df[type][test]["label"], metric)
    fold_2_auc = score_model(params, feats[type][test], input_df[type][test]["label"], feats[type][train], input_df[type][train]["label"], metric)
    return 0.5 * (fold_1_auc + fold_2_auc)


def fill_objective(train, test, type, feats, labs, scoring_metric):
  def filled_obj(trial):
    return objective(trial, train, test, type, feats, labs, scoring_metric)
  return filled_obj

def main(feature_type = "ref", scoring_metric = "PR", n_trials = 200):
    specified_objective = fill_objective("d1", "d2", feature_type, features, input_df, scoring_metric)
    study = optuna.create_study(direction="maximize")
    study.optimize(specified_objective, n_trials = n_trials)
    return(study)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna optimization of hyperparameters.")
    parser.add_argument("--scoring_metric", type=str, default= "PR",
                        help="Full path to directory with labeled examples. ROC, PR, accuracy.")
    parser.add_argument("--feature_type", type=str, default= "ref",
                        help="Mapping of aa representation between mutant and reference. ref.")
    parser.add_argument("--n_trials", type=int, default=200, help="Number of models for oputuna to train.")
    parser.add_argument("--plot_suffix", type=str, default= "PRd1d2ref",
                        help="Name of study to annotate plots.")
    args = parser.parse_args()

    optuna_run = main(args.feature_type, args.scoring_metric, n_trials = args.n_trials)
    plot_optuna_results(optuna_run, args.plot_suffix)



# python3 run_optuna_sklearn.py
# python3 run_optuna_sklearn.py --scoring_metric ROC
