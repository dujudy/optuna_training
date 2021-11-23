
"""Optuna optimization of hyperparameters: Configuration Function

Configuration file for the run_optuna library. Returns dicts specifying:
    - paths to relevant training/testing data
    - start index that partitions metadata (left) from features (right)
    - ensembl protein IDs to exclude from each dataset.
Data must  separate metadata (left) from features (right) at index specified by
'start'.

Also returns lists that specifies the column names of the features and relevant
metadata. Data must include the following columns: protein_id, protein_position,
reference_aa, mutant_aa, and label.

  Typical usage examples:
    from run_optuna_args import *
    root, exclude, ref_paths, mut_paths, pca_mats, start, cols, metas = configure(args)
"""
# CONFIGuration function

def configure(args):
    # Define data folder. Folder must end with "/" character.
    root = "transformed/"

    # Set up genes to exclude from each dataset.
        # To include all genes, use empty list.
    exclude = {
    #    "dataset_1_name": ["Gene 1", "Gene 2", ...],
    #    "dataset_2_name": []
         "d1": [], "d2": [], "mcf10A": [],
    }

    # Define paths to reference and mutant features.
        # If any features DNE, use empty dict.
    mut_paths = {
    "d1": root + "test_d1_primateNegativesDocmPositives_key_Rostlab_Bert_reference.tsv",
    "d2": root + "test_d2_primateNegativesDocmPositives_key_Rostlab_Bert_reference.tsv",
    "mcf10A": root + "test_mmc2_newlabs_key_Rostlab_Bert_reference.tsv"
    }
    ref_paths = {
    }

    # Define paths to PCA matrices applied to data
        # If PCA matrices DNE, use empty dict.
    pca_mats = {
     "pca100": root + 'hs/pca1500aa/pcamatrix_Hg38samp_dim_100.pca',
     "pca250": root + 'hs/pca1500aa/pcamatrix_Hg38samp_dim_250.pca',
     "pca500": root + 'hs/pca1500aa/pcamatrix_Hg38samp_dim500.pca',
     "pca1000": root + 'hs/pca1500aa/pcamatrix_Hg38samp_dim_1000.pca'
    }

    # Define indices for the start of data/end of metadata.
    start = {"d1":6, "d2":6, "t1":8, "t2":8, "mcf10A":15, "ptenDMS":10, "maveDB":13}

    ## Specify column names of feature columns
    cols = ["Rost_" + str(i) for i in range(0,4096)]

    # Metadata column names of interest
    metas = ['protein_id', 'protein_position', 'reference_aa', 'mutant_aa', 'label']

    return root, exclude, ref_paths, mut_paths, pca_mats, start, cols, metas
