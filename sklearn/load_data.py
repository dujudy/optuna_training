import pandas as pd
import numpy as np

def process_data(path, features_start, exclude):
 input_data = pd.read_csv(path, header = 0, sep='\t', comment='#').dropna().reset_index(drop = True)
 input_data = input_data[input_data["protein_id"].isin(exclude) == False]
 input_data['label'] = input_data["label"].replace("positives", 1).replace("negatives", 0)
 features = input_data.iloc[:,features_start:(len(input_data.columns))].to_numpy()
 labels = input_data["label"]
 return input_data, features, labels

### Combine split datasets (t or d)
def combine_d(data_dict):
    return np.concatenate((data_dict["d1"], data_dict["d2"]), axis=0)

def load_data(dataset, tranformation):
    return input_df[tranformation][dataset]

def load_data(ref_paths, mut_paths, start, cols, exclude):
    ## initialize data dicts
    features = {"ref":{}, "mut":{}, "abs":{}, "mutref":{}}#"absref":{}, "absmut":{}
    labels = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
    input_df = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
    ## initialize metadata dicts
    metas = ['protein_id', 'protein_position', 'reference_aa', 'mutant_aa', 'label']
    metadata = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
    # initialize column names
    refcols = [i + "_ref" for i in cols]
    mutcols = [i + "_mut" for i in cols]
    feature_columns = {"ref":{}, "mut":{}, "abs":{}, "absref":{}, "absmut":{}, "mutref":{}};
    feature_columns["ref"] = refcols; feature_columns["mut"] = mutcols

    for key in ref_paths.keys():
        print(key)
        # reading reference + mutant data
        input_df["ref"][key], features["ref"][key], labels["ref"][key] = process_data(ref_paths[key], start[key], exclude[key])
        input_df["mut"][key], features["mut"][key], labels["mut"][key] = process_data(mut_paths[key], start[key], exclude[key])

        metadata["ref"][key] = input_df["ref"][key].iloc[:,0:start[key]]
        metadata["mut"][key] = input_df["mut"][key].iloc[:,0:start[key]]
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
        features[f]["d"] = combine_d(features[f])
        labels[f]["d"] = combine_d(labels[f])
        metadata[f]["d"] = metadata[f]["d1"].append(metadata[f]["d2"]).reset_index(drop = True)

    return features, labels, input_df, metadata, feature_columns
