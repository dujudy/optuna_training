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
