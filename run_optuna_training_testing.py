"""Optuna optimization of hyperparameters.

Implements optuna optimization algorithms using sklearn ML algorithms (currently,
GradientBoostingClassifier.) auROC, auPRC or accuracy between datasets can be
used as the optimization objective. Returns an optuna study class.

  Typical usage examples:
    python3 run_optuna_training_testing.py
    python3 run_optuna_training_testing.py --scoring_metric ROC
"""
from joblib import dump, load
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
from sklearn.model_selection import GroupShuffleSplit

from run_optuna_args import *
from optuna_via_sklearn.config import *
from optuna_via_sklearn.load_data import *
from optuna_via_sklearn.generate_prediction_probs import *
import optuna_via_sklearn.specify_sklearn_models
from optuna_via_sklearn.plot_optuna_results import *

def train_model(model_name, parameters, train_feats, train_labs, save = False):
    # Define model type
    classifier = optuna_via_sklearn.specify_sklearn_models.define_model(model_name, parameters)
    # Train model
    classifier.fit(train_feats, train_labs)
    # Save model
    if save != False:
        dump(classifier, save)
    return classifier

def fill_objective(split, feats, labs, input_df, scoring_metric, model_name):
  def filled_obj(trial):
    return optuna_via_sklearn.specify_sklearn_models.objective(trial, split, feats, labs, input_df, scoring_metric, model_name)
  return filled_obj

def optimize_hyperparams(split, scoring_metric, n_trials,  model_name):
    specified_objective = fill_objective(split, features, labels, input_df, scoring_metric, model_name)
    study = optuna.create_study(direction="maximize")
    study.optimize(specified_objective, n_trials = n_trials)
    return(study)

def score_model(parameters, train_feats, train_labs, test_feats, test_labs, test_gene_id, metric, model_name):
    # Generate prediction probs for test set
    classifier = train_model(model_name, parameters, train_feats, train_labs)
    y_score = classifier.predict_proba(test_feats)[:,1]
    # Generate scoring metric
    if metric == "auPRC": # Calculate auPRC
        precision, recall, thresholds = precision_recall_curve(test_labs, y_score)
        score = auc(recall, precision)
    elif metric == "auROC": # Calculate auROC
        score = roc_auc_score(test_labs, y_score)
    elif metric == "auROC_bygene": # Calculate by-gene auROC
        input_df["ref"]["d1"].groupby('protein_id').apply(lambda x: roc_auc_score(x.label, y_score))
    elif metric == "accuracy": # Calculate mean accuracy
        score = classifier.score(test_feats, test_labs)
    return(score)

if __name__ == "__main__":

    args = parse_run_optuna_args()

    # Load training/testing data
    config = DataSpecification(args)
    features, labels, input_df, metadata = load_data(config)
    print(features)

    # Training Set (Cross-validation) Split
    gss = GroupShuffleSplit(n_splits=1, train_size = 1 / 2, random_state = 5)
    split = gss.split(input_df["training"], groups = input_df["training"]["protein_id"])
    for train, test in split:
         print(input_df["training"]["label"][np.r_[train]].value_counts())
         print(input_df["training"]["label"][np.r_[test]].value_counts())

    input_df["CV1"] = input_df["training"].iloc[train,]
    input_df["CV2"] = input_df["training"].iloc[test,]
    features["CV1"] = features["training"][np.r_[train]]
    features["CV2"] = features["training"][np.r_[test]] 
    labels["CV1"] = labels["training"][np.r_[train]]
    labels["CV2"] = labels["training"][np.r_[test]]  

    # Load PCA if applicable
    if args.pca_key != "None":
        print("Loading PCA matrix at: " + args.pca_key)
        pca = PCA(args.pca_key, config)
        # pca = faiss.read_VectorTransform(config.pca_mats[args.pca_key])

    # Split mutref if applicable
    for data_name in input_df.keys():
        mut =  features[data_name][:,0:len(config.cols)]
        ref =  features[data_name][:,len(config.cols):features[data_name].shape[1]]
        # Apply PCA if applicable
        if args.pca_key != "None":
            mut = pca.apply_pca(mut); ref = pca.apply_pca(ref)
            #features[data_name] = pca.apply_py(np.ascontiguousarray(features[data_name].astype('float32')))
            #mut = pca.apply_py(np.ascontiguousarray(mut.astype('float32')))
            #ref = pca.apply_py(np.ascontiguousarray(ref.astype('float32')))
        if "mutref" in args.training_alias:
            features[data_name] = np.concatenate([mut,ref], axis = 1)
        elif "abs" in args.training_alias:
            features[data_name] = np.abs(mut, ref)
        elif "mutant" in args.training_alias:
            features[data_name] =  mut
        elif "reference" in args.training_alias:
            features[data_name] =  ref
    # Merge PCA-Transformed Bert features and chemical features if applicable
    if "merge" in args.training_alias:
        # Load chemical data to merge with existing features
        merge_data, merge_features, merge_labels = process_data(args.merge_training_path, args.merge_training_start, args.merge_exclude)
        # Establish feature columns
        cols = [i for i in input_df["training"].columns[args.training_start:input_df["training"].shape[1]]]
        for i in merge_data.columns[args.merge_training_start:merge_data.shape[1]]:
            cols.append(i)
        # Merge datasets
        for dataset in input_df.keys():
            if dataset in args.testing_alias:
                merge_testing_data, merge_testing_features, merge_testing_labels = process_data(args.merge_testing_path, args.merge_testing_start, args.merge_exclude)
                merge = input_df[dataset].merge(merge_testing_data, how = "left").dropna().drop_duplicates(subset = ["Gene", "protein_position", "reference_aa", "mutant_aa", "label"])
            else:
                merge = input_df[dataset].merge(merge_data, how = "left").dropna().drop_duplicates(subset = ["Gene", "protein_position", "reference_aa", "mutant_aa", "label"])
            # Update dicts with merged datasets
            input_df[dataset] = merge
            features[dataset] = merge[cols]
            labels[dataset] = merge["label"]
            metadata[dataset] = merge[[i for i in input_df[dataset] if i not in cols]]
            del merge

    # Define prefix for all files produced by run
    run_id = args.results_folder + "/" + "{write_type}" + args.lang_model_type + "-" + args.pca_key + "-" + args.model_name + "-" + args.scoring_metric

    # Check if optuna-trained model already exists
    model_path = run_id.format(write_type=args.training_alias) + '_model.joblib'
    if exists(model_path):
        # Load model
        print("Loading model at: " + model_path)
        final_classifier = load(model_path)
    else:
        # Optimize hyperparameters with optuna
        print("Running optuna optimization.")
        optuna_run = optimize_hyperparams(2, args.scoring_metric, args.n, args.model_name)
        plot_optuna_results(optuna_run, run_id.format(write_type="optuna-" + args.training_alias + "-"))

        # train final model using optuna's best hyperparameters
        final_classifier = train_model(
            args.model_name,
            optuna_run.best_trial.params,
            features["training"],
            labels["training"],
            save = model_path)

    # generate prediction probabilities
    for data_name in config.data_paths:
        if data_name not in ["training", "CV1", "CV2"]:
            print(data_name); print(model_path);
            generate_prediction_probs(final_classifier,
                                      args.model_name + "_opt" + args.scoring_metric,
                                      features[data_name],
                                      labels[data_name],
                                      metadata[data_name],
                                      run_id.format(write_type=data_name + "-" + args.training_alias + "-")
                                     )

    # generate performance plots
    plot_script = "test_ml_models/plot_analyses/"
    for f in ["plot_mcf10A_optuna_comparisons.R", "plot_mavedb_optuna_comparisons.R"]:
        print(f)
        system(plot_script + f)

