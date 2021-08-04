from config import *
import pandas as pd
import numpy as np

def process_data(path, features_start, exclude):
 input_data = pd.read_csv(path, header = 0, sep='\t', comment='#').dropna().reset_index(drop = True)
 input_data = input_data[input_data["protein_id"].isin(exclude) == False]
 input_data['label'] = input_data["label"].replace("positives", 1).replace("negatives", 0)
 features = input_data.iloc[:,features_start:(len(input_data.columns))].to_numpy()
 labels = input_data["label"]
 #np.array([1 if lbl == 'positives' else 0 for lbl in input_data['label'].tolist()])
 return input_data, features, labels

## initialize data dicts
features = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
labels = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
input_df = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
metadata = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
feature_columns = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
feature_columns["ref"] = cols

for key in ref_paths.keys():
    print(key)
    # reading reference + mutant data
    if key in ["ptenDMS", "mcf10A", "maveDB"]:
        input_df["ref"][key], features["ref"][key], labels["ref"][key] = process_data(root + ref_paths[key], start[key], [])
        input_df["mut"][key], features["mut"][key], labels["mut"][key] = process_data(root + mut_paths[key], start[key], [])
    #elif key in ["maveDB"]:
    #    input_df["ref"][key], features["ref"][key], labels["ref"][key] = process_data(root + ref_paths[key], start[key], exclude_maveDB)
    #    input_df["mut"][key], features["mut"][key], labels["mut"][key] = process_data(root + mut_paths[key], start[key], exclude_maveDB)
    else:
        input_df["ref"][key], features["ref"][key], labels["ref"][key] = process_data(root + ref_paths[key], start[key], exclude)
        input_df["mut"][key], features["mut"][key], labels["mut"][key] = process_data(root + mut_paths[key], start[key], exclude)
    metadata["ref"][key] = input_df["ref"][key].iloc[:,0:start[key]]
    metadata["mut"][key] = input_df["mut"][key].iloc[:,0:start[key]]
    metas = list(metadata["mut"][key].columns)
    dim = input_df["ref"][key].shape[1]

    # create mutref
    mutref = input_df["ref"][key].copy().merge(input_df["mut"][key].copy(), on = metas, how = 'left', suffixes = ["", "_ref"]).dropna()
    features["mutref"][key] = mutref.iloc[:,start[key]:mutref.shape[1]].to_numpy()
    #labels["mutref"][key] = np.array([1 if lbl == 'positives' else 0 for lbl in mutref['label'].tolist()])
    labels["mutref"][key] = mutref['label'].tolist()
    metadata["mutref"][key] = mutref.iloc[:,0:start[key]]
    input_df["mutref"][key] = mutref

    # create abs
    features["abs"][key] = mutref.copy()[cols] - mutref.copy()[refcols].to_numpy()
    features["abs"][key] = features["abs"][key].abs().to_numpy()
    #labels["abs"][key] = np.array([1 if lbl == 'positives' else 0 for lbl in mutref['label'].tolist()])
    labels["abs"][key] = mutref['label'].tolist()
    metadata["abs"][key] = mutref.iloc[:,0:start[key]]

    # create absref
    abs = mutref.copy()
    abs[cols] = features["abs"][key]  #replace mut data with abs data
    abs = abs[metas + cols] #create df with only abs data + metadata
    absref = abs.copy()
    absref = absref.iloc[:,0:dim].merge(input_df["ref"][key].copy(), on = metas, how = 'left', suffixes = ["", "_ref"]).dropna()
    features["absref"][key] = absref[cols + refcols].to_numpy()
    #labels["absref"][key] = np.array([1 if lbl == 'positives' else 0 for lbl in absref['label'].tolist()])
    labels["absref"][key] = mutref['label'].tolist()
    metadata["absref"][key] = absref.iloc[:,0:start[key]]

    # create absmut
    absmut = abs.copy()
    absmut = absmut.iloc[:,0:dim].merge(input_df["mut"][key].copy(), on = metas, how = 'left', suffixes = ["", "_mut"]).dropna()
    features["absmut"][key] = absmut[cols + mutcols].to_numpy()
    labels["absmut"][key] = np.array([1 if lbl == 'positives' else 0 for lbl in absmut['label'].tolist()])
    metadata["absmut"][key] = absmut.iloc[:,0:start[key]]
    del mutref, absref, absmut

### Combine split datasets (t or d)
def combine_d(data_dict):
    return np.concatenate((data_dict["d1"], data_dict["d2"]), axis=0)

for f in features.keys():
    features[f]["d"] = combine_d(features[f])
    labels[f]["d"] = combine_d(labels[f])
    metadata[f]["d"] = metadata[f]["d1"].append(metadata[f]["d2"])

def load_data(dataset, tranformation):
    return input_df[tranformation][dataset]
