# Preprocess the data
# split the preprocessed data
# save it in data/processed folder
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params

def preprocess(config_path):
    config = read_params(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    data = pd.read_csv(raw_data_path, sep=",")

    # replacing zero values with the mean of the column
    data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())
    data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].mean())
    data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].mean())
    data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].mean())
    data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].mean())

    #Handling Skeweness
    q = data['Pregnancies'].quantile(0.98)
    # we are removing the top 2% data from the Pregnancies column
    data_cleaned = data[data['Pregnancies'] < q]
    q = data_cleaned['BMI'].quantile(0.99)
    # we are removing the top 1% data from the BMI column
    data_cleaned = data_cleaned[data_cleaned['BMI'] < q]
    q = data_cleaned['SkinThickness'].quantile(0.99)
    # we are removing the top 1% data from the SkinThickness column
    data_cleaned = data_cleaned[data_cleaned['SkinThickness'] < q]
    q = data_cleaned['Insulin'].quantile(0.95)
    # we are removing the top 5% data from the Insulin column
    data_cleaned = data_cleaned[data_cleaned['Insulin'] < q]
    q = data_cleaned['DiabetesPedigreeFunction'].quantile(0.99)
    # we are removing the top 1% data from the DiabetesPedigreeFunction column
    data_cleaned = data_cleaned[data_cleaned['DiabetesPedigreeFunction'] < q]
    q = data_cleaned['Age'].quantile(0.99)
    # we are removing the top 1% data from the Age column
    data_cleaned = data_cleaned[data_cleaned['Age'] < q]
    print(data_cleaned.head())
    return config, data_cleaned

def split_and_saved_data(config_path):
    config, df = preprocess(config_path)
    #config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    #raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    #df = pd.read_csv(raw_data_path, sep=",")
    train, test = train_test_split(
        df,
        test_size=split_ratio,
        random_state=random_state
        )
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)