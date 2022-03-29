"""Optuna optimization of hyperparameters: data loading module

Reads in and processes training/testing data for the optuna_training library.
Returns objects of class dict that contain features or metadata of origin specified by keys.

  Typical usage examples:
    from optuna_via_sklearn.load_data import *
    features, labels, input_df, metadata, feature_columns = load_data(ref_paths, mut_paths, start, cols, exclude, metas, args.feature_type)
"""
import pandas as pd
import numpy as np
import faiss
import pickle
from sklearn.preprocessing import Normalizer

def process_data(path, features_start, exclude):
    # Reads in and processes training/testing data
    print("Loading " + path)
    input_data = pd.read_csv(path, header = 0, sep='\t', comment='#').dropna()

    # Remove ensembl protein IDs specified by exclude
    input_data = input_data[input_data["protein_id"].isin(exclude) == False]

    # Remove cases where mutant amino acids match the reference amino acids
    input_data = input_data[input_data["reference_aa"] != input_data["mutant_aa"]].reset_index(drop = True)

    # Binarize objective label column
    input_data['label'] = input_data["label"].replace("positives", 1).replace("negatives", 0)\

    # Drop Duplicate columns
    input_data = input_data.drop_duplicates().reset_index(drop = True)

    # Subset features and labels
    features = input_data.iloc[:,features_start:(len(input_data.columns))].to_numpy()
    labels = input_data["label"]

    # L2 Normalizer
    L2_features = Normalizer(norm = "l2").fit_transform(features)

    return input_data, L2_features, labels

#def load_data(ref_paths, mut_paths, start, cols, exclude, metas, feat_type):
def load_data(config):
    # Initialize data dicts
    features = {}; labels = {}; input_df = {}; metadata = {};
    
    # Load Data
    for key in config.data_paths.keys():
        print(key)
        # Reading in data
        input_df[key], features[key], labels[key] = process_data(config.data_paths[key], config.start[key], config.exclude[key])
        metadata[key] = input_df[key].iloc[:,0:config.start[key]]

    return features, labels, input_df, metadata

class PCA:
    def __init__(self, pca_key, config):
        self.key = pca_key
        self.path = config.pca_mats[self.key]
        if "claire" in pca_key:
            self.handle = open(self.path, "rb")
            self.pca = pickle.load(self.handle)
        else:
            self.pca = faiss.read_VectorTransform(self.path)

    def apply_pca(self, input):
        if "claire" in self.key:
            output = input @ self.pca["pcamatrix"].T + self.pca["bias"]
        else:
            output = self.pca.apply_py(np.ascontiguousarray(input.astype('float32')))
        return(output)
