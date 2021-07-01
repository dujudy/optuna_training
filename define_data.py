import tensorflow as tf

def tensor_data(feats, labs):
    def data_fn():
        X = feats
        y = labs
        samples = tf.data.Dataset.from_tensor_slices((X.values, y))
        return samples
    return data_fn
