from model import ExplainableModel
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

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = get_data()

    params = {"objective": "binary:logistic",
              "verbosity": 1,
              "tree_method": "gpu_hist"}

    model = ExplainableModel(x_train, y_train, params=params,
                             path="./weights/model.pkl",
                             explainer_path="./weights/explainer.pkl",
                             seed=42)

    # model.train(iterations=300, do_train=False, load=True, save=False)

    y_hat = model.predict(x_test)

    print("F1: " + str(f1_score(y_test, y_hat)))
    print("Precision: " + str(precision_score(y_test, y_hat)))
    print("Recall: " + str(recall_score(y_test, y_hat)))
    print("Accuracy: " + str(accuracy_score(y_test, y_hat)))

    print("Explain Plot")
    model.explain(x_test.iloc[0])
