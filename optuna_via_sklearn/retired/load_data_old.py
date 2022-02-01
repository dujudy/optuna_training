"""Optuna optimization of hyperparameters: data loading module

Reads in and processes training/testing data for the optuna_training library.
Returns objects of class dict that contain features or metadata of origin specified by keys.

  Typical usage examples:
    from optuna_via_sklearn.load_data import *
    features, labels, input_df, metadata, feature_columns = load_data(ref_paths, mut_paths, start, cols, exclude, metas, args.feature_type)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

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

def load_data(ref_paths, mut_paths, start, cols, exclude, metas, feat_type, split):
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
            if split != None:
                split = 1 / float(split)
                input_df[f]["crossvalidation_1"], input_df[f]["crossvalidation_1"] = train_test_split(input_df[f]["training"], train_size=split, random_state=7, shuffle=True, stratify="protein_id")
                metadata[f]["crossvalidation_1"] = input_df[f]["crossvalidation_1"].iloc[:,0:start["training"]]
                metadata[f]["crossvalidation_2"] = input_df[f]["crossvalidation_2"].iloc[:,0:start["training"]]
                features[f]["crossvalidation_1"] = input_df[f]["crossvalidation_1"].iloc[:,start:(len(input_df[f]["crossvalidation_1"].columns))].to_numpy()
                features[f]["crossvalidation_2"] = input_df[f]["crossvalidation_2"].iloc[:,start:(len(input_df[f]["crossvalidation_2"].columns))].to_numpy()
                labels[f]["crossvalidation_1"] = input_df[f]["crossvalidation_1"]["label"]
                labels[f]["crossvalidation_2"] = input_df[f]["crossvalidation_2"]["label"]
            else:
                features[f]["training"] = combine_d(features[f])
                labels[f]["training"] = combine_d(labels[f])
                metadata[f]["training"] = metadata[f]["d1"].append(metadata[f]["d2"]).reset_index(drop = True)
    return features, labels, input_df, metadata, feature_columns

"""Optuna optimization of hyperparameters: Configuration Function

Configuration file for the run_optuna library. Returns dicts specifying:
    - paths to relevant training/testing data
    - start index that partitions metadata (left) from features (right)
    - ensembl protein IDs to exclude from each dataset.
Data must  separate metadata (left) from features (right) at index specified by
'start'.

Returns lists that specifies the column names of the features and relevant
metadata.

All data must be tab-separated files with metadata (left) fully partitioned from
the features (right). Moreover, all data must include the following columns:
    protein_id, protein_position, reference_aa, mutant_aa, and label.
"""

def configure(args):
    # define training data folder
    root = "/tigress/jtdu/map_language_models/user_files/"
    scratch = "/scratch/gpfs/jtdu/primateNegatives_uniq_key_mut/"

    # set up exclude genes to exclude from each dataset.
    exclude = {"d1": [], "d2": [], "training":[],
        "mcf10A":["ENSP00000361021", "ENSP00000483066"], #PTEN
        "maveDB": ["ENSP00000312236", "ENSP00000350283", "ENSP00000418960", "ENSP00000417148", "ENSP00000496570", "ENSP00000465818", "ENSP00000467329", "ENSP00000465347", "ENSP00000418775", "ENSP00000418548", "ENSP00000420705", "ENSP00000419481", "ENSP00000420412", "ENSP00000418819", "ENSP00000418212", "ENSP00000417241", "ENSP00000326002", "ENSP00000489431", "ENSP00000397145", "ENSP00000419274", "ENSP00000498906", "ENSP00000419988", "ENSP00000420253", "ENSP00000418986", "ENSP00000419103", "ENSP00000420201", "ENSP00000495897", "ENSP00000417554", "ENSP00000417988", "ENSP00000420781", "ENSP00000494614", "ENSP00000478114"] #BRCA1
    }

    if args.lang_model_type == "Rostlab_Bert":
        ## Define paths to Bert Vectors of reference sequences
        ref_paths = {
        "d1": root + "d1_primateNegativesDocmPositives_key_Rostlab_Bert_reference.tsv",
        "d2": root + "d2_primateNegativesDocmPositives_key_Rostlab_Bert_reference.tsv",
        "mcf10A": root + "mmc2_newlabs_key_Rostlab_Bert_reference.tsv",
        "maveDB": root + "mavedb_offset_key_Rostlab_Bert_reference.tsv"
        }

        ## Define paths to Bert Vectors of mutant sequences
        mut_paths = {
        "training": scratch + "primateNegatives_uniq_key_clinvar_maxlen_Rostlab_Bert_mutant_MSKImpact.tsv",
        "d1": root + 'd1_primateNegativesDocmPositives_key_Rostlab_Bert_mutant.tsv',
         "d2": root + 'd2_primateNegativesDocmPositives_key_Rostlab_Bert_mutant.tsv',
        "mcf10A": root + "mmc2_newlabs_key_Rostlab_Bert_mutant.tsv",
        "maveDB": root + "mavedb_offset_key_Rostlab_Bert_mutant.tsv"
        }

        ## Define paths to PCA matrices applied to data
        pca_mats = {
         "pca100": root + 'hs/pca1500aa/pcamatrix_Hg38samp_dim_100.pca',
         "pca250": root + 'hs/pca1500aa/pcamatrix_Hg38samp_dim_250.pca',
         "pca500": root + 'hs/pca1500aa/pcamatrix_Hg38samp_dim_500.pca',
         "pca1000": root + 'hs/pca1500aa/pcamatrix_Hg38samp_dim_1000.pca'
        }

        ## define indices for start of data/end of metadata
        start = {"d1":6, "d2":6, "t1":8, "t2":8, "mcf10A":15, "ptenDMS":10, "maveDB":13, "training":5}

        ## specify column names
        cols = ["Rost_" + str(i) for i in range(0,4096)]

    elif args.lang_model_type == "UniRep":
        ## define paths to UniRep Vectors of reference sequences
        ref_paths = {
            "d1": root + "d1_primateNegativesDocmPositives_key_UniRep_reference.tsv", #"d1_primateNegativesDocmPositives_reference.tsv",
            "d2": root + "d2_primateNegativesDocmPositives_key_UniRep_reference.tsv", #"d2_primateNegativesDocmPositives_reference.tsv",
            #"mcf10A": root + "mmc2_newlabs_key_UniRep_reference.tsv"
            "maveDB": "/scratch/gpfs/jtdu/mavedb_offset_key_UniRep_reference.tsv"
        }

        ## Define paths to UniRep Vectors of mutant sequences
        mut_paths = {
         "d1": root + 'd1_primateNegativesDocmPositives_key_UniRep_mutant.tsv',
         "d2": root + 'd2_primateNegativesDocmPositives_key_UniRep_mutant.tsv',
         #"mcf10A": root + "mmc2_newlabs_key_UniRep_mutant.tsv"
         "maveDB": "/scratch/gpfs/jtdu/mavedb_offset_key_UniRep_mutant.tsv"
        }

        ## Define paths to PCA matrices applied to data
        pca_mats = {

        }

        ## Define indices for start of data/end of metadata
        # data must separate data (on right) from metadata (left)
        start = {"d1":5, "d2":5, "t1":8, "t2":8, "mcf10A":14, "ptenDMS":10, "maveDB":12 }

        ## Specify column names
        cols = ["UR_" + str(i) for i in range(0,1900)]

    # metadata column names of interest
    metas = ['protein_id', 'protein_position', 'reference_aa', 'mutant_aa', 'label']

    return root, exclude, ref_paths, mut_paths, pca_mats, start, cols, metas
