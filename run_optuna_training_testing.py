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

from run_optuna_args import *
from optuna_via_sklearn.config_function import *
from optuna_via_sklearn.generate_prediction_probs import *
from optuna_via_sklearn.load_data import *
from optuna_via_sklearn.plot_optuna_results import *
import optuna_via_sklearn.specify_sklearn_models

def train_model(model_name, parameters, train_feats, train_labs, save = False):
    # Define model type
    classifier = optuna_via_sklearn.specify_sklearn_models.define_model(model_name, parameters)
    # Train model
    classifier.fit(train_feats, train_labs)
    # Save model
    if save != False:
        dump(classifier, save + '_model.joblib')
    return classifier

def fill_objective(train, test, type, feats, labs, input_df, scoring_metric, model_name):
  def filled_obj(trial):
    return optuna_via_sklearn.specify_sklearn_models.objective(trial, train, test, type, feats, labs, input_df, scoring_metric, model_name)
  return filled_obj

def optimize_hyperparams(feature_type, scoring_metric, n_trials,  model_name):
    specified_objective = fill_objective("crossvalidation_1", "crossvalidation_2", feature_type, features, labels, input_df, scoring_metric,  model_name)
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
        score = 0; n = 0
        for gene in test_gene_id.unique():
            # subset labels by protein id
            idx = test_gene_id == gene
            subset = test_labs[idx]
            if len(subset.unique()) > 1:
                # subset labels by protein id
                n += 1
                score += roc_auc_score(subset, y_score[idx])
            score = score / n

        score = roc_auc_score(test_labs, y_score)
    elif metric == "auROC_bygene": # Calculate by-gene auROC

    input_df["ref"]["d1"].groupby('protein_id').apply(lambda x: roc_auc_score(x.label, y_score))

    elif metric == "accuracy": # Calculate mean accuracy
        score = classifier.score(test_feats, test_labs)
    return(score)

if __name__ == "__main__":

    args = parse_run_optuna_args()

    # Load training/testing data
    root, exclude, ref_paths, mut_paths, pca_mats, start, cols, metas = configure(args)
    features, labels, input_df, metadata, feature_columns = load_data(ref_paths, mut_paths, start, cols, exclude, metas, args.feature_type)

    # Apply PCA if applicable
    if args.pca_key != "None":
        print("Loading PCA matrix at: " + args.pca_key)
        pca = faiss.read_VectorTransform(pca_mats[args.pca_key])
        newfeat = args.feature_type + "_" + args.pca_key; features[newfeat] = {}; labels[newfeat] = {}; metadata[newfeat] = {};

        for data_name in features[args.feature_type].keys():
            features[newfeat][data_name] = pca.apply_py(np.ascontiguousarray(features[args.feature_type][data_name].astype('float32')))
            labels[newfeat][data_name] = labels[args.feature_type][data_name]
            metadata[newfeat][data_name] = metadata[args.feature_type][data_name]
            run_id = args.results_folder + "/" + args.pca_key.replace(".pkl", "") + "_"+ args.lang_model_type + args.feature_type + ".png"
            if data_name == "mut":
                plot_pca(features[newfeat][data_name], metadata[newfeat][data_name]["mutant_aa"], output_name)
            elif data_name == "ref":
                plot_pca(features[newfeat][data_name], metadata[newfeat][data_name]["reference_aa"], output_name)
            del data_name
        args.feature_type = newfeat

    # Define prefix for all files produced by run
    run_id = args.results_folder + "/" + "{write_type}" + args.
    model_type + "_" + args.feature_type + "_" + args.model_name + args.scoring_metric
    # Check if optuna-trained model already exists
    model_path = run_id.format(write_type="full_") + '_model.joblib'
    if exists(model_path):
        # Load model
        print("Loading model at: " + model_path)
        final_classifier = load(model_path)
    else:
        # Optimize hyperparameters with optuna
        print("Running optuna optimization.")
        optuna_run = optimize_hyperparams(args.feature_type, args.scoring_metric, args.n, args.model_name)
        plot_optuna_results(optuna_run, run_id.format(write_type="optuna_"))

        # train final model using optuna's best hyperparameters
        final_classifier = train_model(
            args.model_name,
            optuna_run.best_trial.params,
            features[args.feature_type]["training"],
            labels[args.feature_type]["training"],
            save = model_path)

    # generate prediction probabilities
    for data_name in ref_paths:
        if data_name not in ["crossvalidation_1","crossvalidation_2", "training"]:
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
