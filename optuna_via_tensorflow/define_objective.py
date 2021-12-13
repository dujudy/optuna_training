from define_data import *
from define_gbcmodel import *

EPOCHS = 10

def objective(trial, train, test, type, input_df, labs, fnames):
    # Define Feature column names
    feats = fnames[type]
    #
    # Generate model
    model = define_model(trial, feats)
#
    # Train model
        #for epoch in range(EPOCHS):
    model.train(tensor_data(input_df[type][train][feats], input_df[type][train]["label"]), max_steps=100)
#
    # Eval model
    results = model.evaluate(tensor_data(input_df[type][test][feats], input_df[type][test]["label"]))
    #trial.report(results.accuracy, epoch)
#
    clear_output()
#
    # Handle pruning
    #if trial.should_prune():
    #    raise optuna.exceptions.TrialPruned()
#
    return results.accuracy


def fill_objective(train, test, type, feats, labs, fnames):
  def filled_obj(trial):
    return objective(trial, train, test, type, feats, labs, fnames)
  return filled_obj
