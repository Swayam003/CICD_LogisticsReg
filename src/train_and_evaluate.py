# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from get_data import read_params
import argparse
import joblib
import json


def train_and_evaluate(config_path):
    config = read_params(config_path)
    x_test_data_path = config["split_data"]["x_test_path"]
    y_test_data_path = config["split_data"]["y_test_path"]
    x_train_data_path = config["split_data"]["x_train_path"]
    y_train_data_path = config["split_data"]["y_train_path"]
    model_dir = config["model_dir"]

    x_train = pd.read_csv(x_train_data_path, sep=",")
    y_train = pd.read_csv(y_train_data_path, sep=",")
    x_test = pd.read_csv(x_test_data_path, sep=",")

    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    predicted_qualities = log_reg.predict(x_test)
    print(predicted_qualities)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(log_reg, model_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)