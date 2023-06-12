from explanationplots import *
from model import ExplainableModel
import pandas as pd
from sklearn.model_selection import train_test_split
from explanationneighbors import *
from explanationcounterfactuals import *
from explanationtexts import *



feature_description = {
    "Transaktionswert": "text",
     "alter Kontostand Sender": "text",
    "neuer Kontostand Sender": "text",
    "alter Kontostand Empfänger": "text",
    "neuer Kontostand Empfänger": "text",
}

feature_names = {
    "isFraud": "isFraud",
    "amount": "Transaktionswert",
    "oldbalanceOrg": "alter Kontostand Sender",
    "newbalanceOrig": "neuer Kontostand Sender",
    "oldbalanceDest": "alter Kontostand Empfänger",
    "newbalanceDest": "neuer Kontostand Empfänger",
}

continuous_variables = [feature_names["amount"], feature_names["oldbalanceOrg"], feature_names["newbalanceOrig"],
                        feature_names["oldbalanceDest"], feature_names["newbalanceDest"]]

feat_names = [feature_names["amount"], feature_names["oldbalanceOrg"], feature_names["newbalanceOrig"],
                        feature_names["oldbalanceDest"], feature_names["newbalanceDest"]]
def get_data():

    df = pd.read_csv("data/paysim.csv")
    df = df.drop("isFlaggedFraud", axis=1)

    df = df.drop(['nameOrig', 'nameDest', 'type', 'step'], axis=1)

    df = df.reset_index().rename(columns=feature_names)

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
                             feature_names=feature_names,
                             path="./weights/model.pkl",
                             continuous_variables=continuous_variables,
                             explainer_path="./weights/explainer.pkl",
                             seed=42)

    # model.train(iterations=300, do_train=True, load=False, save=True)

    y_hat = model.predict(x_test)
    # model.explain(x_test.iloc[0])

    df_test = x_test
    df_test["label"] = y_hat
    df_fraud = df_test[df_test["label"] == 1]
    df_fraud = df_fraud.drop(['label'], axis=1)

    fig1 = create_feature_importance_plot(model, df_fraud.iloc[0], feat_names)
    fig2 = create_class_cluster(model, df_fraud.iloc[0])
    fig3 = create_detailed_feature_plot(model, df_fraud.iloc[0], 0, feature_names["amount"], x_max=200000)
    # neighbors = get_n_neighbors_information(model, df_fraud.iloc[0], feature_names, n_neighbors=3)
    # counterfactuals = get_n_counterfactuals(model, df_fraud[0:1], n_factuals=4)
    text = create_explanation_texts(model, df_fraud[0:1], 1, feat_names, feature_description)

    print(text)
    # fig1.show()
    # fig2.show()
    # fig3.show()



