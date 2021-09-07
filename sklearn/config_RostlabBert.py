# CONFIGuration file

## define training data folder
root = "/tigress/jtdu/map_language_models/user_files/"

# set up exclude genes
exclude = {"d1": [], "d2": [],
    "mcf10A":["ENSP00000361021", "ENSP00000483066"], #PTEN
    "maveDB": ["ENSP00000312236", "ENSP00000350283", "ENSP00000418960", "ENSP00000417148", "ENSP00000496570", "ENSP00000465818", "ENSP00000467329", "ENSP00000465347", "ENSP00000418775", "ENSP00000418548", "ENSP00000420705", "ENSP00000419481", "ENSP00000420412", "ENSP00000418819", "ENSP00000418212", "ENSP00000417241", "ENSP00000326002", "ENSP00000489431", "ENSP00000397145", "ENSP00000419274", "ENSP00000498906", "ENSP00000419988", "ENSP00000420253", "ENSP00000418986", "ENSP00000419103", "ENSP00000420201", "ENSP00000495897", "ENSP00000417554", "ENSP00000417988", "ENSP00000420781", "ENSP00000494614", "ENSP00000478114"] #BRCA1
}
## define paths to Bert Vectors of reference sequences
ref_paths = {
"d1": root + "d1_primateNegativesDocmPositives_key_Rostlab_Bert_reference.tsv",
"d2": root + "d2_primateNegativesDocmPositives_key_Rostlab_Bert_reference.tsv",
"mcf10A": root + "mmc2_newlabs_key_Rostlab_Bert_reference.tsv",
#"maveDB": "/scratch/gpfs/jtdu/mavedb_offset_key_Rostlab_Bert_reference.tsv"
}

## define paths to UniRep Vectors of mutant sequences
mut_paths = {
"d1": root + 'd1_primateNegativesDocmPositives_key_Rostlab_Bert_mutant.tsv',
 "d2": root + 'd2_primateNegativesDocmPositives_key_Rostlab_Bert_mutant.tsv',
 "mcf10A": root + "mmc2_newlabs_key_Rostlab_Bert_mutant.tsv"
 #"maveDB": "/scratch/gpfs/jtdu/mavedb_offset_key_Rostlab_Bert_mutant.tsv"
}

## define indices for start of data/end of metadata
# data must separate data (on right) from metadata (left)
start = {"d1":5, "d2":5, "t1":8, "t2":8, "mcf10A":14, "ptenDMS":10, "maveDB":11}

## specify column names
cols = ["Rost_" + str(i) for i in range(0,1024)] 
metas = ['protein_id', 'protein_position', 'reference_aa', 'mutant_aa', 'label']
