# CONFIGuration file

## define training data folder
root = "/tigress/jtdu/optuna_training/complete_datasets/"

# set up exclude genes
exclude = ["ENSP00000361021", "ENSP00000483066"] #PTEN
exclude_maveDB = ["ENSP00000312236", "ENSP00000350283", "ENSP00000418960", "ENSP00000417148", "ENSP00000496570", "ENSP00000465818", "ENSP00000467329", "ENSP00000465347", "ENSP00000418775", "ENSP00000418548", "ENSP00000420705", "ENSP00000419481", "ENSP00000420412", "ENSP00000418819", "ENSP00000418212", "ENSP00000417241", "ENSP00000326002", "ENSP00000489431", "ENSP00000397145", "ENSP00000419274", "ENSP00000498906", "ENSP00000419988", "ENSP00000420253", "ENSP00000418986", "ENSP00000419103", "ENSP00000420201", "ENSP00000495897", "ENSP00000417554", "ENSP00000417988", "ENSP00000420781", "ENSP00000494614", "ENSP00000478114"] #BRCA1


## define paths to UniRep Vectors of reference sequences
 # data must include the columns protein_id, protein_position, reference_aa, mutant_aa, and label
ref_paths = {
"d1":"d1_primateNegativesDocmPositives_reference.tsv",
"d2":"d2_primateNegativesDocmPositives_reference.tsv",
#"t1":"t1_pooledTCGA_new_reference.tsv",
#"t2":"t2_pooledTCGA_new_reference.tsv",
#"mcf10A":"mmc2_new_reference.tsv",
#"ptenDMS":"PTEN_DMS_phosph_reference.tsv",
"mcf10A": "mmc2_new_reference_missenseOnly.tsv",
#"maveDB": "mavedb_offset_reference.tsv"
}

## define paths to UniRep Vectors of mutant sequences
mut_paths = {
"d1":'d1_primateNegativesDocmPositives_mutant.tsv',
 "d2":'d2_primateNegativesDocmPositives_mutant.tsv',
 #"t1":"t1_pooledTCGA_new_mutant.tsv",
 #"t2":'t2_pooledTCGA_mutant_new.tsv',
 #"mcf10A":'mmc2_new_mutant.tsv',
 "mcf10A": "mmc2_new_mutant_missenseOnly.tsv",
 "ptenDMS":"PTEN_DMS_phosph_mutant.tsv",
 "maveDB": "mavedb_offset_mutant.tsv"
}

## define indices for start of data/end of metadata
# data must separate data (on right) from metadata (left)
start = {"d1":5, "d2":5, "t1":8, "t2":8, "mcf10A":13, "ptenDMS":10, "maveDB":11 }

## specify column names
cols = ["UR_" + str(i) for i in range(0,1900)]
refcols = ["UR_" + str(i)+ "_ref" for i in range(0,1900)]
mutcols = ["UR_" + str(i)+ "_mut" for i in range(0,1900)]
metas = ['protein_id', 'protein_position', 'reference_aa', 'mutant_aa', 'label']
