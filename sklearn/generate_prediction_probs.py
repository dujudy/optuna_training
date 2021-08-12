"""Testing of Machine Learning Model.

Tests optuna-optimized algorithms on a given external dataset.
Returns the corresponding prediction probabilities.

  Typical usage examples:
    python3 generate_prediction_probs.py --model_path results/d_auROC_GB_model.joblib --testing_pfix results/d_auROC_GB_mmc
"""
# testing: generate prediction probabilities

def generate_prediction_probs(model, model_name,X, y, metas, testing_pfix):
  probs = model.predict_proba(X)

  # compile pred_probs data to save
  pp_dataframe = pd.DataFrame({"prediction_prob": probs[:,1], "label": y,
                               "models":np.repeat(model_name, probs.shape[0])})
  pp_dataframe = pd.concat([metas[['protein_id', 'protein_position','reference_aa', 'mutant_aa']],pp_dataframe], axis = 1)  
  pp_dataframe.to_csv(path_or_buf = "./" + testing_pfix + "_PredictionProbs.csv")
  return(pp_dataframe)

if __name__ == "__main__":
    import argparse
    from joblib import load
    from sklearn.ensemble import GradientBoostingClassifier
    from load_data import *

    parser = argparse.ArgumentParser(description="Test ML Models: Generate prediction probabilities of test set")
    parser.add_argument("--model_path", type=str,
                        help="Full path to directory to trained ML model.")
    parser.add_argument("--testing_pfix", type=str, default= "test",
                        help="Prefix for filename: prediction probabilities.")
    parser.add_argument("--feature_type", type=str, default= "ref",
                        choices = ["ref", "mut"],
                        help="Mapping of aa representation between mutant and reference. ref.")
    parser.add_argument("--model_name", type = str, default = "GB", choices = ["GB"],
                        help="Name of Machine Learning algorithm.")
    args = parser.parse_args()

    for data_name in ref_paths:
        if data_name not in ["d1","d2", "d"]:
            print(data_name)
            model = load(args.model_path)
            generate_prediction_probs(model, args.model_name,
                                      features[args.feature_type][data_name],
                                      labels[args.feature_type][data_name],
                                      metadata[args.feature_type][data_name], args.testing_pfix)
