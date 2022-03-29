"""Optuna optimization of hyperparameters: Configuration Function

Specifies paths to relevant training data. Returns an optuna study class.
=======
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
class DataSpecification():
    def __init__(self, args):
        if args.lang_model_type == "Rostlab_Bert":
            ## Define path dict to training/testing data
            self.data_paths = {
                "training": args.training_path,
                args.testing_alias: args.testing_path
#                "mcf10A": "/tigress/jtdu/map_language_models/user_files/mmc2_newlabs_key_Rostlab_Bert_mutant.tsv",
#                "maveDB": "/tigress/jtdu/map_language_models/user_files/mavedb_offset_key_Rostlab_Bert_mutant.tsv"
            }
            ## Define indices for start of data/end of metadata
            self.start = {"d1":6, "d2":6, "t1":8, "t2":8, "mcf10A":15, "ptenDMS":10, "maveDB":13, "training":args.training_start, args.testing_alias:args.testing_start}

#            if args.testing_path is not None:
#                for i in range(0, len(args.testing_path)):
#                    self.data_paths[args.testing_alias[i]] = args.testing_path[i]
#                    self.start[args.testing_alias[i]] = args.testing_start[i]

            ## Define path dict to PCA matrices applied to data
            self.pca_mats = {
             "pca100": "/tigress/jtdu/map_language_models/user_files/hs/pca_samp10000/pcamatrix_Hg38samp_dim_100.pca",
             "pca250": "/tigress/jtdu/map_language_models/user_files/hs/pca_samp10000/pcamatrix_Hg38samp_dim_250.pca",
             "pca500": "/tigress/jtdu/map_language_models/user_files/hs/pca_samp10000/pcamatrix_Hg38samp_dim_500.pca",
             "pca1000": "/tigress/jtdu/map_language_models/user_files/hs/pca_samp10000/pcamatrix_Hg38samp_dim_1000.pca",
             "pca128claire": "/tigress/jtdu/map_language_models/user_files/claire_pca/qfo_sample5000.fasta.aa.128dim.pcamatrix.pkl",
             "pca256claire": "/tigress/jtdu/map_language_models/user_files/claire_pca/qfo_sample5000.fasta.aa.256dim.pcamatrix.pkl",
             "pca512claire": "/tigress/jtdu/map_language_models/user_files/claire_pca/qfo_sample5000.fasta.aa.512dim.pcamatrix.pkl"
            }

            ## specify column names
            self.cols = ["Rost_" + str(i) for i in range(0,4096)]

        elif args.lang_model_type == "UniRep":
            ## define paths to UniRep Vectors of reference sequences
            self.data_paths = {
                "d1_ref": root + "d1_primateNegativesDocmPositives_key_UniRep_reference.tsv", #"d1_primateNegativesDocmPositives_reference.tsv",
                "d2_ref": root + "d2_primateNegativesDocmPositives_key_UniRep_reference.tsv", #"d2_primateNegativesDocmPositives_reference.tsv",
                "mcf10A_ref": root + "mmc2_newlabs_key_UniRep_reference.tsv",
                "maveDB_ref": "/scratch/gpfs/jtdu/mavedb_offset_key_UniRep_reference.tsv",
                 "d1_mut": root + 'd1_primateNegativesDocmPositives_key_UniRep_mutant.tsv',
                 "d2_mut": root + 'd2_primateNegativesDocmPositives_key_UniRep_mutant.tsv',
                 "mcf10A_mut": root + "mmc2_newlabs_key_UniRep_mutant.tsv",
                 "maveDB_mut": "/scratch/gpfs/jtdu/mavedb_offset_key_UniRep_mutant.tsv"
            }

            ## Define indices for start of data/end of metadata
            self.start = {"d1":5, "d2":5, "t1":8, "t2":8, "mcf10A":14, "ptenDMS":10, "maveDB":12 }

            ## Define paths to PCA matrices applied to data
            self.pca_mats = {
            }

            ## Specify column names
            self.cols = ["UR_" + str(i) for i in range(0,1900)]

        # metadata column names of interest
        self.metas = ['protein_id', 'protein_position', 'reference_aa', 'mutant_aa', 'label']
        # set up exclude genes to exclude from each dataset.
        self.exclude = {
            "training": [], "testing": [], args.testing_alias:[],
            "mcf10A":[], #["ENSP00000361021", "ENSP00000483066"], #PTEN
            "maveDB": ["ENSP00000312236", "ENSP00000350283", "ENSP00000418960", "ENSP00000417148", "ENSP00000496570", "ENSP00000465818", "ENSP00000467329", "ENSP00000465347", "ENSP00000418775", "ENSP00000418548", "ENSP00000420705", "ENSP00000419481", "ENSP00000420412", "ENSP00000418819", "ENSP00000418212", "ENSP00000417241", "ENSP00000326002", "ENSP00000489431", "ENSP00000397145", "ENSP00000419274", "ENSP00000498906",
                       "ENSP00000419988", "ENSP00000420253", "ENSP00000418986", "ENSP00000419103", "ENSP00000420201", "ENSP00000495897", "ENSP00000417554", "ENSP00000417988", "ENSP00000420781", "ENSP00000494614", "ENSP00000478114"] #BRCA1
        }
