"""Optuna optimization of hyperparameters: data loading module

Reads in and processes training/testing data for the optuna_training library.
Returns objects of class dict that contain features or metadata of origin specified by keys.

  Typical usage examples:
    from optuna_via_sklearn.load_data import *
    features, labels, input_df, metadata, feature_columns = load_data(ref_paths, mut_paths, start, cols, exclude, metas, args.feature_type)
"""
import pandas as pd
import numpy as np

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

    return input_data, features, labels

### Combine split datasets
def combine_d(data_dict):
    return np.concatenate((data_dict["crossvalidation_1"], data_dict["crossvalidation_2"]), axis=0)

def load_data(ref_paths, mut_paths, start, cols, exclude, metas, feat_type):
    # Initialize data dicts
    features = {"ref":{}, "mut":{}, "abs":{}, "mutref":{}}#"absref":{}, "absmut":{}
    labels = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
    input_df = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};

    # Initialize metadata dicts
    metadata = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};

    # Initialize column names
    refcols = [i + "_ref" for i in cols]
    mutcols = [i + "_mut" for i in cols]
    feature_columns = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
    feature_columns["ref"] = refcols; feature_columns["mut"] = mutcols

    for key in mut_paths.keys():
        print(key)
        if feat_type != "mut":
            # Reading in reference data
            input_df["ref"][key], features["ref"][key], labels["ref"][key] = process_data(ref_paths[key], start[key], exclude[key])
            metadata["ref"][key] = input_df["ref"][key].iloc[:,0:start[key]]
            if feat_type == "ref":
                pass
        if feat_type != "ref":
            # Reading in mutant data
            input_df["mut"][key], features["mut"][key], labels["mut"][key] = process_data(mut_paths[key], start[key], exclude[key])
            metadata["mut"][key] = input_df["mut"][key].iloc[:,0:start[key]]
            if feat_type == "mut":
                pass
        if feat_type not in ["ref", "mut"]:
            # Further processing necessary for feature_type abs, mutref
            metas = list(metadata["mut"][key].columns)
            dim = input_df["ref"][key].shape[1]

            # create mutref
            mutref = input_df["ref"][key].copy().merge(input_df["mut"][key].copy(), on = metas, how = 'left', suffixes = ["", "_ref"]).dropna()
            features["mutref"][key] = mutref.iloc[:,start[key]:mutref.shape[1]].to_numpy()
            #labels["mutref"][key] = np.array([1 if lbl == 'positives' else 0 for lbl in mutref['label'].tolist()])
            labels["mutref"][key] = mutref['label']#.tolist()
            metadata["mutref"][key] = mutref.iloc[:,0:start[key]]
            input_df["mutref"][key] = mutref

            # create abs
            features["abs"][key] = mutref.copy()[cols] - mutref.copy()[refcols].to_numpy()
            features["abs"][key] = features["abs"][key].abs().to_numpy()
            #labels["abs"][key] = np.array([1 if lbl == 'positives' else 0 for lbl in mutref['label'].tolist()])
            labels["abs"][key] = mutref['label']#.tolist()
            metadata["abs"][key] = mutref.iloc[:,0:start[key]]

    for f in features.keys():
        if len(features[f]) > 0:
            features[f]["training"] = combine_d(features[f])
            labels[f]["training"] = combine_d(labels[f])
            metadata[f]["training"] = metadata[f]["crossvalidation_1"].append(metadata[f]["crossvalidation_2"]).reset_index(drop = True)
    return features, labels, input_df, metadata, feature_columns
