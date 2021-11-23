"""Optuna optimization of hyperparameters: PCA tranformation module

Reads in specified PCA matrix and applies transformation to data.
Returns objects of class dict that contain transformed features or metadata of origin specified by keys.

  Typical usage examples:
    from optuna_via_sklearn.plot_pca import *
     = load_data(ref_paths, mut_paths, start, cols, exclude, metas, args.feature_type)
"""
import faiss

def plot_pca(pca_mat, aa_col, output_name):
    # Plots the first two principal components

    pca_df = pd.DataFrame({"PC1": pca_mat[:,0], "PC2": pca_mat[:,1], "amino_acid":aa_col})
    fig = sns.scatterplot(x = "PC1", y = "PC2", hue = "amino_acid", data = pca_df).get_figure()
    fig.savefig(output_name)
    
def run_optuna_pcatransform(args):
    # Reads in specified PCA matrix and applies transformation to data.
    pca = faiss.read_VectorTransform(pca_mats[args.pca_key])

    # Initialize new feature dicts
    newfeat = args.feature_type + "_" + args.pca_key; features[newfeat] = {}; labels[newfeat] = {};

    for data_name in features[args.feature_type].keys():
        if len(features[feature])
        features[newfeat][data_name] = pca.apply_py(np.ascontiguousarray(features[args.feature_type][data_name].astype('float32')))
        labels[newfeat][data_name] = labels[args.feature_type][data_name]
        metadata[newfeat][data_name] = metadata[args.feature_type][data_name]
        run_id = args.results_folder + "/" + args.pca_key.replace(".pkl", "") + "_"+ args.lang_model_type + args.feature_type + ".png"
        if data_name == "mut":
            plot_pca(features[newfeat][data_name], metadata[newfeat][data_name]["mutant_aa"], output_name)
        elif data_name == "ref":
            plot_pca(features[newfeat][data_name], metadata[newfeat][data_name]["reference_aa"], output_name)
        del data_name
    return()
