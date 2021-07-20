"""
Optuna optimization
"""
import argparse
import optuna
from define_objective import *

import optuna
from load_data import *
from define_objective import *


specified_objective = fill_objective("d1", "d2", "ref", input_df, labels, feature_columns)


study = optuna.create_study(direction="maximize")
study.optimize(specified_objective, n_trials=25)







def run_optuna(survey = True):
    study = optuna.create_study(direction="maximize")
    if args.survey:
        study.optimize(lambda trial: objective(trial, "d1", "d2", "ref"), n_trials=25)
    else:
        study.optimize(lambda trial: objective(trial, "d1", "d2", "ref"), n_trials=100)

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run Optuna Optimization")
    parser.add_argument("survey", type=bool, default = False)
    args = parser.parse_args()

    run_optuna(args.survey)
