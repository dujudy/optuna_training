from run_optuna_sklearn import *
from optuna.visualization import plot_param_importances, plot_parallel_coordinate, plot_slice, plot_optimization_history

def plot_optuna_results(study, study_name = "PRd1d2ref"):
    print(study)
    fig = plot_optimization_history(study)
    fig.write_image(study_name + "_optina_optimizationhistory.png")

    fig = plot_parallel_coordinate(study)
    fig.write_image(study_name + "_optuna_parallelcoordinate.png")

    fig = plot_slice(study)
    fig.write_image(study_name + "_optuna_slice.png")

    fig = plot_param_importances(study)
    fig.write_image(study_name + "_optuna_paramimportances.png")
    return(fig)
