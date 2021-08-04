# testing: generate prediction probabilities

def generate_prediction_probs(model_path, model_name, X, y, data_name):

  model = load(model_path)
  probs = model.predict_proba(X)

  # compile pred_probs data to save
  pp_dataframe = pd.DataFrame({"prediction_prob": probs[:,1], "label": y})
  size = pp_dataframe.shape[0]
  pp_dataframe = pd.concat([pd.DataFrame({"models":np.repeat(model_name, size)}),
                            pp_dataframe], axis = 1)

  pp_dataframe.to_csv(path_or_buf = "./" + model_name + "/" + data_name + "_PredictionProbs.csv")
  return(pp_dataframe)

if __name__ == "__main__":
    import argparse
    from sklearn.ensemble import GradientBoostingClassifier
    from load_data import *

    parser = argparse.ArgumentParser(description="Test ML Models: Generate prediction probabilities of test set")
    parser.add_argument("--model_path", type=str,
                        help="Full path to directory to trained ML model.")
    parser.add_argument("--model_name", type = str, default = "GB", choices = ["GB"],
                         help="Name of Machine Learning algorithm.")
    args = parser.parse_args()

    for data_name in ref_paths:
        if data_name not in ["d1","d2", "d"]:
            print(data_name)
            generate_prediction_probs(args.model_path, args.model_name,
                                      features[args.feature_type]["d"],
                                      labels[args.feature_type]["d"], data_name)
