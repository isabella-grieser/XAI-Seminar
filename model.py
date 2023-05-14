import xgboost as xgb
import shap
import numpy as np
import pickle
import pandas as pd


class ExplainableModel:

    def __init__(self, x, y, params,
                 path="./weights/default.pkl",
                 explainer_path="./weights/explainer.pkl",
                 seed=42):
        self.x = x
        self.y = y
        self.params = params
        self.seed = seed
        self.path = path
        self.explainer_path = explainer_path
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
        if save:
            with open(self.path, 'wb') as f:
                pickle.dump(self. model, f)
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

    def explain(self, x):
        explain_val = x.to_numpy().reshape(1, -1)
        y = self.predict(explain_val)

        shap_values = self.explainer.shap_values(explain_val)
        feature_importances = np.abs(shap_values).mean(0)

        print("Y is " + str(y))
        feature_zip = sorted(zip(self.x.columns, feature_importances), key=lambda t: t[1], reverse=True)
        for c, v in feature_zip:
            print(f"{c}: {v}")
