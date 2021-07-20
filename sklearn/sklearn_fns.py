import os, sys, argparse, sklearn
import numpy as np
import pandas as pd
#import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupShuffleSplit
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0}) # This silences the warning messages during plotting
##############################################################################################################
# CONSTANTS
##############################################################################################################
# These are the same names as the module imports, more can be added, just add it to this list and add another if-block to get_model()
MODEL_NAMES = ["RandomForestClassifier", "AdaBoostClassifier", "GradientBoostingClassifier",
               "KNeighborsClassifier", "Logistic"] #"XGBoost"]
DATASETS = ["validation", "test"]

#Hyperparameters for classifiers
N_ESTIMATORS = 10
N_JOBS = 32
N_NEIGHBORS =  10

#############################################################################################################
# HELPER FUNCTIONS
#############################################################################################################

def get_model(model_name):
    '''
    Returns instances of machine learning models from Python modules
    :param model_name: Exact name of model's module in Python
    :return: an instance of the specified model
    '''
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Found unknown model name: {model_name}")
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                  n_jobs=N_JOBS)
    elif model_name == "AdaBoostClassifier":
        return AdaBoostClassifier(n_estimators=N_ESTIMATORS)
    elif model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier(n_estimators=N_ESTIMATORS)
    elif model_name == "KNeighborsClassifier":
        return KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS)
    elif model_name == "Logistic":
        return LogisticRegression(random_state=10)
    #elif model_name == "XGBoost":
    #  return xgb.XGBClassifier()

def evaluate_accuracy(pred_lbls, true_lbls):
    """
        Evaluate performance by comparing predictions to ground truth.
        Returns a dictionary of various result metrics.
    """
    ret = {}
    correct = 0
    for i in range(0, len(pred_lbls)):
        if pred_lbls[i] == true_lbls[i]:
            correct += 1
    ret["accuracy"] = float(correct) / len(pred_lbls)
    return ret

def predict_data(model_dict, name, dataset, data, lbls):
    """
        Uses model to labels for data and evaluates the results
         by calculating model accuracy, prediction probability, precision, and recall.
    """
    model_dict[name][dataset]["predictions"] = model_dict[name]["model"].predict(data)
    model_dict[name][dataset]["performance"] = evaluate_accuracy(model_dict[name][dataset]["predictions"], lbls)
    model_dict[name][dataset]["prediction_prob"] = model_dict[name]["model"].predict_proba(data)
    model_dict[name][dataset]["labels"] = lbls
    return model_dict

def analyze_predictions(model_dict, name, dataset, lbls):
    """
        Check the quality of model predictions by calculating precision, recall, confusion matrix,
        Returns an updated dictionary with the above metrics.
    """
    #if "predictionProb" not in model_dict[name][dataset]:
    #    print("Error: no prediction probabilities incorprated into metrics")
        # what's a better way to flag an error?

    model_dict[name][dataset]["precision"], model_dict[name][dataset]["recall"], thresh = precision_recall_curve(y_true = lbls,
                       probas_pred = model_dict[name][dataset]["prediction_prob"][:,1])
    model_dict[name][dataset]["fpr"], model_dict[name][dataset]["tpr"], thresh  = roc_curve(y_true = lbls, y_score = model_dict[name][dataset]["prediction_prob"][:,1])
    model_dict[name][dataset]["confusion_matrix"] = confusion_matrix(lbls, model_dict[name][dataset]["predictions"])
    model_dict[name][dataset]["performance"]["roc_auc"] = roc_auc_score(y_true = lbls, y_score = model_dict[name][dataset]["prediction_prob"][:,1])
    model_dict[name][dataset]["performance"]["TP"] = model_dict[name][dataset]["confusion_matrix"][1,1]
    model_dict[name][dataset]["performance"]["TN"] = model_dict[name][dataset]["confusion_matrix"][0,0]
    model_dict[name][dataset]["performance"]["FP"] = model_dict[name][dataset]["confusion_matrix"][0,1]
    model_dict[name][dataset]["performance"]["FN"] = model_dict[name][dataset]["confusion_matrix"][1,0]

    return model_dict

def print_performance(model_dict, name, dataset):
    """
        Prints model.
    """
    print("%s: \"%s\"" % (dataset.title(), name))
    for metric_name, metric_value in iter(sorted(model_dict[name][dataset]["performance"].items())):
        print("\t%s:%s" % (metric_name, metric_value))
    return model_dict


#def compile_performance_dataframes(model_dict, name, fold, input_df, features_start):
def compile_performance_dataframes(model_dict, name, fold):
    perf_dataframe = pd.concat([pd.DataFrame({"models": name, "fold": fold}, [1]),
                                      pd.DataFrame(model_dict[name]["test"]["performance"], [1])], axis=1)
    size = model_dict[name]["test"]["recall"].shape[0]
    pr_dataframe = pd.concat([pd.DataFrame({"models": np.repeat(name, size), "fold": np.repeat(fold, size)}),
                        pd.DataFrame({
                                     "Precision": model_dict[name]["test"]["precision"],
                                     "Recall": model_dict[name]["test"]["recall"]
                                     })], axis=1)
    size = model_dict[name]["test"]["fpr"].shape[0]
    roc_dataframe = pd.concat([pd.DataFrame({"models": np.repeat(name, size), "fold": np.repeat(fold, size)}),
                        pd.DataFrame({
                                     "TPR": model_dict[name]["test"]["tpr"],
                                     "FPR": model_dict[name]["test"]["fpr"]
                                     })], axis=1)
    ## feature importances
#    size = len(input_df.columns) - features_start
#    fi_dataframe = pd.concat([pd.DataFrame({"models": np.repeat(name, size), "fold": np.repeat(fold, size)}),
#                        pd.DataFrame({
#                                     "features": model_dict[name]["feature_importance"],
#                                     "importance": input_df.columns[features_start:(len(input_df.columns) )]
#                                     })], axis=1)

    ## Prediction Probabilities
    ## validation set
    ppv_dataframe = pd.DataFrame({"prediction_prob": model_dict[name]["validation"]["prediction_prob"][:,1], "label": model_dict[name]["validation"]["labels"]})
    size = ppv_dataframe.shape[0]
    ppv_dataframe = pd.concat([pd.DataFrame({"models": np.repeat(name, size), "fold": np.repeat(fold, size)}),
                              ppv_dataframe], axis = 1)
    ## testing set
    ## Prediction Probabilities
    ppt_dataframe = pd.DataFrame({"prediction_prob": model_dict[name]["test"]["prediction_prob"][:,1], "label": model_dict[name]["test"]["labels"]})
    size = ppt_dataframe.shape[0]
    ppt_dataframe = pd.concat([pd.DataFrame({"models": np.repeat(name, size), "fold": np.repeat(fold, size)}),
                              ppt_dataframe], axis = 1)
    return perf_dataframe, pr_dataframe, roc_dataframe, ppv_dataframe, ppt_dataframe


#, fi_dataframe

def save_analysis_plots(models, model_name, output_folder, fold, data, lbls):
    """
    Plots PR Curves and AUPRC curves for the True and Shuffled Testing labels, then saves them in the same folder.
    Output: none
    """


    sns.relplot(y = "Precision", x = "Recall", kind = "line",
                data = pd.DataFrame({"Precision":models[model_name]["test"]["precision"],
                                     "Recall":models[model_name]["test"]["recall"]})
               ).savefig("./" + output_folder + "/PRCurve_" + model_name + str(fold) + ".png")

    sns.relplot(y = "TPR", x = "FPR", kind = "line",
                data = pd.DataFrame({"TPR":models[model_name]["test"]["tpr"],
                                     "FPR":models[model_name]["test"]["fpr"]})
               ).savefig("./" + output_folder + "/ROCurve_" + model_name + str(fold) + ".png")

    false_preds = predict_data(models, model_name, "test", data, shuffle(lbls, random_state = 7))
    false_preds = analyze_predictions(models, model_name, "test", shuffle(lbls, random_state = 7))

    curr_performance, curr_pr, curr_roc, curr_fi, curr_probs = compile_performance_dataframes(models, model_name, fold, input_df, features_start)

    sns.relplot(y = "Precision", x = "Recall", kind = "line",
                data = pd.DataFrame({"Precision":false_preds[model_name]["test"]["precision"],
                                     "Recall":false_preds[model_name]["test"]["recall"]}), color = "red"
               ).savefig("./" + output_folder + "/PRCurveShuffled_" + model_name + str(fold) + ".png")
    sns.relplot(y = "TPR", x = "FPR", kind = "line",
                data = pd.DataFrame({"TPR":false_preds[model_name]["test"]["tpr"],
                                     "FPR":false_preds[model_name]["test"]["fpr"]}), color = "red"
               ).savefig("./" + output_folder + "/ROCurveShuffled_" + model_name + str(fold) + ".png")
