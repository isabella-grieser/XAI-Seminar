import xgboost as xgb
import shap
import numpy as np
import pickle


class ExplainableModel:

    def __init__(self, dtrain, dtest, params,
                 path="./weights/default.model",
                 explainer_path="./weights/explainer.pkl",
                 seed=42):
        self.dtrain = dtrain
        self.dtest = dtest
        self.params = params
        self.seed = seed
        self.path = path
        self.explainer_path = explainer_path
        self.model = None
        self.explainer = None

    def train(self, iterations=300, early_stopping_rounds=10, do_train=True, load=False, save=True):
        if load:
            self.model = xgb.Booster()
            self.model.load_model(self.path)
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
        if do_train:
            evallist = [(self.dtrain, 'train'), (self.dtest, 'eval')]
            self.model = xgb.train(self.params,
                                   self.dtrain,
                                   iterations,
                                   evallist,
                                   early_stopping_rounds=early_stopping_rounds)
            self.explainer = shap.TreeExplainer(self.model)
        if save:
            self.model.save_model(self.path)
            with open(self.explainer_path, 'wb') as f:
                pickle.dump(self.explainer, f)

    def predict(self, x):
        if self.model is None:
            self.model = xgb.Booster()
            self.model.load_model(self.path)
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
        return [round(value) for value in self.model.predict(x)]

    def predict_proba(self, x):
        if self.model is None:
            self.model = xgb.Booster()
            self.model.load_model(self.path)
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
        return self.model.predict(x)

    def explain(self, x):
        y = self.predict(x)

        shap_values = self.explainer.shap_values(x)
        feature_importances = np.abs(shap_values).mean(0)

        print("Y is " + str(y))
        print(feature_importances)
