import optuna
from load_data import *
import tensorflow as tf
from os import getcwd

def tensor_data(feats, labs):
    def data_fn():
        X = feats
        y = labs
        samples = tf.data.Dataset.from_tensor_slices((X.values, y))
        return samples
    return data_fn


def define_model(trial, feature_names):
    params = {
      'n_trees': trial.suggest_int("n_trees", 25, 150),
      'max_depth': trial.suggest_int("max_depth", 1, 8),
      "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.25),
      "l1_regularization": trial.suggest_uniform("l1_regularization", 0,1),
      "tree_complexity": trial.suggest_uniform("tree_complexity", 0.0, .25),
      "min_node_weight": trial.suggest_uniform("min_node_weight", 0.0, .25),
      "quantile_sketch_epsilon": trial.suggest_uniform("quantile_sketch_epsilon", 0.0, .25),
      "pruning_mode": trial.suggest_categorical("pruning_mode",	["none", "pre", "pos"])
    }
    fname_tensors = []
    for f in feature_names:
        fname_tensors.append(tf.feature_column.numeric_column(f))
    model = tf.estimator.BoostedTreesClassifier(
        feature_columns = fname_tensors, model_dir = getcwd(), weight_column=None,
        train_in_memory = True,  n_batches_per_layer = 1, center_bias = True,
        **params
    )
    return model

def objective(trial, train, test, type, input_df, labs, fnames):
    # Define Feature column names
    feats = fnames[type]
    # Generate model
    model = define_model(trial, feats)
    # Train model
        #for epoch in range(EPOCHS):
    model.train(tensor_data(input_df[type][train][feats], labs[type][train]), max_steps=100)
    # Eval model
    results = model.evaluate(tensor_data(input_df[type][test][feats], labs[type][test]))
    #trial.report(results.accuracy, epoch)
    # Handle pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return results.accuracy


def fill_objective(train, test, type, feats, labs, fnames):
  def filled_obj(trial):
    return objective(trial, train, test, type, feats, labs, fnames)
  return filled_obj

specified_objective = fill_objective("d1", "d2", "ref", input_df, labels, feature_columns)
study = optuna.create_study(direction="maximize")
study.optimize(specified_objective, n_trials=5)
