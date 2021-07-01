from define_data import *
from define_gbcmodel import *

EPOCHS = 10

def objective(trial, train, test, type, input_df, labs, fnames):
    # Define Feature column names
    feats = fnames[type]

    # Generate model
    model = define_model(trial, feats)
    print()

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


def test_objective(input_df, labs, fnames, train = "d1", test = "d2", type = "ref"):
    # Define Feature column names
    feats = fnames[type]

    # Train model
        #for epoch in range(EPOCHS):
    print(input_df[type][train][feats])
    test = tensor_data(input_df[type][train][feats], labs[type][train])()
    for f, l in test:
        print(f.shape, l.shape)


    # model.train(tensor_data(input_df[type][train][feats], labs[type][train]), max_steps=100)
    #
    # # Eval model
    # results = model.evaluate(tensor_data(input_df[type][test][feats], labs[type][test]))
    # #trial.report(results.accuracy, epoch)
    #
    # # Handle pruning
    # if trial.should_prune():
    #     raise optuna.exceptions.TrialPruned()
    #
    # return results.accuracy
