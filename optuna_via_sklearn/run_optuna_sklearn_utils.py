import optuna
import pandas as pd
import faiss
import pickle

from os import system
from os.path import exists

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def define_model(model_name, params):
    if model_name =="GB":
        return GradientBoostingClassifier(**params)
    #elif model_name == "SVC": fix
    #elif model_name == "NN": fix
    else:
        raise SomeError("Model name not valid.")

def objective(trial, train, test, type, feats, input_df, metric,  model_name):
    # define model
    if model_name == "GB":
        params = {
            'max_depth': trial.suggest_int("max_depth", 1, 20),
            "n_estimators": trial.suggest_int("n_estimators", 75, 300),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            #"min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 25), # make min larger 1--> 5?
            "random_state": 7}
            ## Unaltered default params
                #loss='deviance', learning_rate=0.1, subsample=1.0, criterion='friedman_mse', min_weight_fraction_leaf=0.0,
                #min_impurity_split=None, max_leaf_nodes=None, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0
                #min_weight_fraction_leaf=0.0
#    elif model_name == "SVC": fix
#    elif model_name == "NN": fix
    else:
        raise SomeError("Model name not valid.")

    # train and evaluate models
    fold_1_auc = score_model(params, feats[type][train], labels[type][train], feats[type][test], labels[type][test], metric,  model_name)
    fold_2_auc = score_model(params, feats[type][test], labels[type][test], feats[type][train], labels[type][train], metric,  model_name)
    return 0.5 * (fold_1_auc + fold_2_auc)
