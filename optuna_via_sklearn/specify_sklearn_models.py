import optuna
import pandas as pd
import faiss
import pickle

from run_optuna_training_testing import *
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def define_model(model_name, params):
    if model_name =="GB":
        return GradientBoostingClassifier(**params)
    elif model_name == "SVC":
        return SVC(probability = True,**params)
    elif model_name == "SVC_balanced":
        return SVC(probability = True,**params)
    elif model_name == "NN":
        return MLPClassifier(**params)
    elif model_name == "Elastic":
        return LogisticRegression(**params)
    else:
        raise SomeError("Model name not valid.")

def objective(trial, split, feats, labs, input_df, metric,  model_name):
    # define model
    if model_name == "GB":
        params = {
            "max_depth": trial.suggest_int("max_depth", 25, 200),
            "n_estimators": trial.suggest_int("n_estimators", 300, 600),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 25), # make min larger 1--> 5?
            "random_state": 7
            }
            ## Unaltered default params
                #loss='deviance', learning_rate=0.1, subsample=1.0, criterion='friedman_mse', min_weight_fraction_leaf=0.0,
                #min_impurity_split=None, max_leaf_nodes=None, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0
                #min_weight_fraction_leaf=0.0
    elif model_name == "SVC":
        params = {
            "C" : trial.suggest_float("C", 1e-3, 1),
            "kernel" : trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
            "random_state": 7
        }
        ## Unaltered default params
            #degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)[source]??
    elif model_name == "SVC_balanced":
        params = {
            "C" : trial.suggest_float("C", 1e-3, 1),
            "kernel" : trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
	    "random_state": 7,
            "class_weight": "balanced"
        }
    elif model_name == "NN":
        params = {
            "hidden_layer_sizes" : (trial.suggest_int("hidden_layer_sizes", 100, 1000)),
            "alpha" : trial.suggest_float("alpha", 5e-6, 5e-4),
            "random_state": 7
        }
        ## Unaltered default params
            #activation='relu', solver='adam',, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000
    elif model_name == "Elastic":
        params = {
            "l1_ratio":trial.suggest_float("l1_ratio", 0, 1),
            "C" : trial.suggest_float("C", 1, 1000),
            "penalty": "elasticnet", "solver":"saga", "max_iter":1000,
            "random_state": 7
        }
    else:
        raise SomeError("Model name not valid.")
    
    # train and evaluate models
    fold_1_auc = score_model(params, feats["CV1"], labs["CV1"], feats["CV2"], labs["CV2"], input_df["CV2"], metric,  model_name)
    fold_2_auc = score_model(params, feats["CV2"], labs["CV2"], feats["CV1"], labs["CV1"], input_df["CV1"], metric,  model_name)
    return 0.5 * (fold_1_auc + fold_2_auc)
