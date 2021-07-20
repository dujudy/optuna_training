from os import getcwd
import tensorflow as tf

def define_model(trial, feature_names):
    params = {
      'n_trees': trial.suggest_int("n_trees", 25, 150),
      # 'max_depth': trial.suggest_int("max_depth", 1, 8),
      # "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.25),
      # "l1_regularization": trial.suggest_uniform("l1_regularization", 0,1),
      # "tree_complexity": trial.suggest_uniform("tree_complexity", 0.0, .25),
      # "min_node_weight": trial.suggest_uniform("min_node_weight", 0.0, .25),
      # "quantile_sketch_epsilon": trial.suggest_uniform("quantile_sketch_epsilon", 0.0, .25),
      # "pruning_mode": trial.suggest_categorical("pruning_mode",	["none", "pre", "pos"])
    }
    fname_tensors = []
    for f in feature_names:
        fname_tensors.append(tf.feature_column.numeric_column(f, dtype = tf.float64))
    model = tf.estimator.BoostedTreesClassifier(
        feature_columns = fname_tensors, model_dir = getcwd(), weight_column=None,
        train_in_memory = True,  n_batches_per_layer = 1, center_bias = True,
        **params
    )
    return model
