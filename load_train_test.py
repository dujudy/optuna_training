import pandas as pd
import numpy as np
import pickle
#from train_models_helpers import *
tf.random.set_seed(007)

# load data -------------------------------------------------------
from load_data import *

# cast model ------------------------------------------------------

from run_optuna import *

objective = objective(trial, training, eval_input_fn)
run_optuna(objective, trial = True)

precomputed weight column
# train model ---------------------------

# test model

# save model + results
