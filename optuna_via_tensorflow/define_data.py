import tensorflow as tf

def tensor_data(feats, labs):
    def data_fn():
        samples = tf.data.Dataset.from_tensor_slices((feats.to_dict(orient='list'), labs)).batch(len(feats))
        return samples
    return data_fn
