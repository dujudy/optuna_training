
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

  Typical usage examples:
    from run_optuna_args import *
    root, exclude, ref_paths, mut_paths, pca_mats, start, cols, metas = configure(args)
"""
# CONFIGuration function

def configure(args):
    # Define data folder. Folder must end with "/" character.
    root = "optuna_via_sklearn/user_files/"

    # Set up genes to exclude from each dataset.
        # To include all genes, use empty list.
    exclude = {
         "crossvalidation_1": [], "crossvalidation_2": [],
        # Add testing sets below.
        # "testing_dataset_1_name": ["Gene 1", "Gene 2", ...],
        # "testing_dataset_2_name": []
    }

    # Define paths to mutant features.
    mut_paths = {
        "crossvalidation_1": root + "file_name.tsv",
        "crossvalidation_2": root + "file_name.tsv",
        # Add testing sets below.
        "testing_dataset_1_name": root + "file_name.tsv"
    }

    # Define paths to reference features.
        # If reference features do not apply, use empty dict.
    ref_paths = {
    }

    # Define paths to PCA matrices applied to data
        # If no PCA transformation is necessary, use empty dict.
    pca_mats = {
    }

    # Define indices for the start of data/end of metadata.
    start = {
        "crossvalidation_1":6,
        "crossvalidation_2":6,
        # Add testing sets below.
        "testing_dataset_1_name":15}

    ## Specify column names of feature columns
    cols = ["Rost_" + str(i) for i in range(0,4096)]

    # Metadata column names of interest
    metas = ['protein_id', 'protein_position', 'reference_aa', 'mutant_aa', 'label']

    return root, exclude, ref_paths, mut_paths, pca_mats, start, cols, metas

