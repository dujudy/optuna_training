"""Testing of Machine Learning Model.

Tests optuna-optimized algorithms on a given external dataset.
Returns the corresponding prediction probabilities.

  Typical usage examples:
    python3 generate_prediction_probs.py --model_path results/d_auROC_GB_model.joblib --testing_pfix results/d_auROC_GB_model
"""
# testing: generate prediction probabilities

def generate_prediction_probs(model_path, model_name, X, y, testing_pfix):

  model = load(model_path)
  probs = model.predict_proba(X)

  # compile pred_probs data to save
  pp_dataframe = pd.DataFrame({"prediction_prob": probs[:,1], "label": y})
  size = pp_dataframe.shape[0]
  pp_dataframe = pd.concat([pd.DataFrame({"models":np.repeat(model_name, size)}),
                            pp_dataframe], axis = 1)

  pp_dataframe.to_csv(path_or_buf = "./" + testing_pfix + "_PredictionProbs.csv")
  return(pp_dataframe)

if __name__ == "__main__":
    import argparse
    from sklearn.ensemble import GradientBoostingClassifier
    from load_data import *

    parser = argparse.ArgumentParser(description="Test ML Models: Generate prediction probabilities of test set")
    parser.add_argument("--model_path", type=str,
                        help="Full path to directory to trained ML model.")
    parser.add_argument("--testing_pfix", type=str, default= "test",
                        help="Prefix for filename: prediction probabilities.")
    args = parser.parse_args()

    for data_name in ref_paths:
        if data_name not in ["d1","d2", "d"]:
            print(data_name)
            generate_prediction_probs(args.model_path, args.model_name,
                                      features[args.feature_type]["d"],
                                      labels[args.feature_type]["d"], testing_pfix)
