from model import ExplainableModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

feature_description = {
    'Number of p peaks missed': 'The R-Peak to P-Peak difference (Amount of R-Peaks - Amount of P-Peaks)',
    'score1': 'The proportion of R-R distances lies inside ±30 limit with respect to average number of samples per heartbeat',
    'score2': 'The proportion of R-R distances lies inside ±30 limit with respect to second quarter of the distribution of the R-peak to R-peak distances',
    'score3': 'The proportion of R-R distances lies inside ±1 standard deviation with respect to second quarter of the distribution of the R-peak to R-peak distances',
    'sd1': 'The short-term heart variability rate',
    'sd2': 'The long-term heart variability rate',
    'ratio': 'The unpredictability of the heartbeat rate',
    'beat_rate': 'The heart rate',
    'dominant_freq': 'The dominant frequency of the signal',
    'energy_percent_at_dominant_freq': 'The energy ratio of the dominant frequency compared to the other frequencies',
    'mean1': 'The mean distance between R-Peaks',
    'std1': 'The variance of the distances between R-peaks',
    'q2_1': 'The second quarter of the distribution of the R-peak to R-peak distances'
    }


def get_data():
    df = pd.read_csv("data/paysim.csv")
    df = df.drop("isFlaggedFraud", axis=1)

    df = df.drop(['nameOrig', 'nameDest'], axis=1)
    df = pd.get_dummies(df, columns=['type'])

    train, test = train_test_split(df, test_size=0.1, stratify=df["isFraud"])
    x_train, y_train = train.drop(['isFraud', 'step'], axis=1), train["isFraud"]
    x_test, y_test = test.drop(['isFraud', 'step'], axis=1), test["isFraud"]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = get_data()

    params = {"objective": "binary:logistic",
              "verbosity": 1,
              "tree_method": "gpu_hist"}

    feature_names = [
        "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
        "type_PAYMENT", "type_TRANSFER"
    ]

    model = ExplainableModel(x_train, y_train, params=params,
                             path="./weights/model.pkl",
                             explainer_path="./weights/explainer.pkl",
                             seed=42)

    model.train(iterations=300, do_train=True, load=False, save=False)

    y_hat = model.predict(x_test)

    print("F1: " + str(f1_score(y_test, y_hat)))
    print("Precision: " + str(precision_score(y_test, y_hat)))
    print("Recall: " + str(recall_score(y_test, y_hat)))
    print("Accuracy: " + str(accuracy_score(y_test, y_hat)))

    print("Explain Plot")
    model.explain(x_test.iloc[0])
