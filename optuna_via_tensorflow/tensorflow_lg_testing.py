from load_data import *
from os import getcwd
import tensorflow as tf

# feature column -----------------
fname_tensors = []
for f in cols:
    fname_tensors.append(tf.feature_column.numeric_column(f))

# data function -----------------

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y)).batch(len(X))
    return dataset
  return input_fn

 # model ------------
model = tf.estimator.BoostedTreesClassifier(
    feature_columns = fname_tensors, model_dir = getcwd(), weight_column=None,
    train_in_memory = True,  n_batches_per_layer = 1, center_bias = True
)

train_input_fn = make_input_fn(dftrain[cols], dftrain["label"])

dftrain = input_df["ref"]["d1"]
dftrain['label'] = dftrain["label"].replace("positives", 1).replace("negatives", 0)

model.train(train_input_fn, max_steps=10)
print(model.evaluate(train_input_fn))
# training -------
linreg = tf.estimator.LinearClassifier(feature_columns=fname_tensors)
linreg.train(train_input_fn, max_steps=10)
print(linreg.evaluate(train_input_fn))
model.t
