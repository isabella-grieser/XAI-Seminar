import xgboost as xgb
import shap
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from explanationplots import *
from explanationtexts import *

class ExplainableModel:

    def __init__(self, x, y, params,
                 feature_names=None,
                 feature_description=None,
                 path="./weights/default.pkl",
                 explainer_path="./weights/explainer.pkl",
                 helper_data_path="./weights/helper_data.csv",
                 seed=42):
        self.x = x
        self.y = y
        self.params = params
        self.seed = seed
        self.path = path
        self.explainer_path = explainer_path
        self.helper_data_path = helper_data_path
        self.feature_names = feature_names
        self.feature_description=feature_description
        self.model = None
        self.explainer = None

    def train(self, iterations=300, early_stopping_rounds=10, do_train=True, load=False, save=True):
        if load:
            self.model = pickle.load(open(self.path, 'rb'))
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
        if do_train:
            self.model = xgb.XGBClassifier()
            self.model.fit(self.x, self.y)
            self.explainer = shap.TreeExplainer(self.model)
            df = self.x
            df['isFraud'] = self.y

            # calculate the medians for the features of the two different classes
            fraud_means = df[df['isFraud'] == 1].median(axis=0)
            normal_means = df[df['isFraud'] == 0].median(axis=0)

            means = pd.concat([fraud_means, normal_means], axis=1).transpose()
            means['label'] = ['Fraud', 'Not Fraud']
            means.to_csv(self.helper_data_path)
        if save:
            with open(self.path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.explainer_path, 'wb') as f:
                pickle.dump(self.explainer, f)

    def predict(self, x):
        if self.model is None:
            self.model = pickle.load(open(self.path, 'rb'))
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
        return [round(value) for value in self.model.predict(x)]

    def predict_proba(self, x):
        if self.model is None:
            self.model = pickle.load(open(self.path, 'rb'))
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
        return self.model.predict(x)

    def explain(self, x, show_feature_amount=5):
        explain_val = x.to_numpy().reshape(1, -1)
        y = self.predict(explain_val)
        y_probs = self.predict_proba(explain_val)[0]
        shap_values = self.explainer.shap_values(explain_val)
        means = pd.read_csv(self.helper_data_path)

        feature_importances = np.abs(shap_values).mean(0)
        # sort by feature importance
        feature_importance = pd.DataFrame(list(zip(self.feature_names, feature_importances)),
                                          columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

        # get the most important feature names
        most_important_feats = feature_importance['col_name'].to_numpy()[:show_feature_amount]

        # get the feature importance percentage
        importance = feature_importance['feature_importance_vals'].to_numpy()
        feature_importance['feature_importance_vals'] = feature_importance['feature_importance_vals'].apply(
            lambda x: x / np.sum(importance) * 100)
        importance = feature_importance['feature_importance_vals'].to_numpy()[:show_feature_amount]

        feature_vec_name = pd.DataFrame(x, columns=self.feature_names)

        plt.figure(0, figsize=(22, 10))

        col = 16
        row = 24

        intro_space = plt.subplot2grid((row, col), (0, 0), colspan=col)
        intro_space.axis('off')

        # for easier access to the relevant features
        feats = dict(zip(feature_importance['col_name'], feature_importance['feature_importance_vals']))

        feature_plot = plt.subplot2grid((row, col), (1, 0), colspan=col // 2, rowspan=row // 2 - 1)

        text_space = plt.subplot2grid((row, col), (row // 2 + 3, 0), colspan=col, rowspan=row // 2 - 2)
        text_space.axis('off')

        self.__create_explanation_intro(intro_space, y, y_probs)
        create_explanation_texts(text_space, most_important_feats, importance, feature_vec_name, y, means,
                                 self.feature_description)

        plt.suptitle("Explanation", fontsize=40)
        plt.show()

        return most_important_feats, importance, shap_values[1][:show_feature_amount]

    def __create_explanation_intro(self, plot, prediction, y_pred_probs):
        txt = f"Predicted Label: {prediction}             Label Probabilities: Fraud: {round(y_pred_probs[0], 2)},  " \
              f"Not Fraud: {round(y_pred_probs[1], 2)}"
        plot.text(0.5, 2, txt, horizontalalignment='left', verticalalignment='center', transform=plot.transAxes,
                  fontsize=25, style='oblique', ha='center',
                  va='top', wrap=True)