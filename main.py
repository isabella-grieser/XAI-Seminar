from model import ExplainableModel
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def get_data():
    df = pd.read_csv("data/paysim.csv")
    df = df.drop("isFlaggedFraud", axis=1)

    df = df.drop(['nameOrig', 'nameDest'], axis=1)
    df = pd.get_dummies(df, columns=['type'])

    train, test = train_test_split(df, test_size=0.1, stratify=df["isFraud"])
    x_train, y_train = train.drop(['isFraud'], axis=1), train["isFraud"]
    x_test, y_test = test.drop(['isFraud'], axis=1), test["isFraud"]

    dtrain_reg = xgb.DMatrix(x_train, y_train, label=y_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(x_test, y_test, label=y_train, enable_categorical=True)

    return dtrain_reg, dtest_reg, y_test


if __name__ == '__main__':
    dtrain_reg, dtest_reg, y_test = get_data()

    params = {"objective": "binary:logistic",
              "verbosity": 2,
              "tree_method": "gpu_hist"}
    model = ExplainableModel(dtrain_reg, dtest_reg, params=params, path="./weights/model", seed=42)
    model.train(iterations=300, do_train=True, load=False, save=True)

    y_hat = model.predict(dtest_reg)

    print(y_hat)
    print("F1: " + str(f1_score(y_test, y_hat)))
    print("Precision: " + str(precision_score(y_test, y_hat)))
    print("Recall: " + str(recall_score(y_test, y_hat)))
    print("Accuracy: " + str(accuracy_score(y_test, y_hat)))
