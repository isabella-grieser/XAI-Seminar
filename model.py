import xgboost as xgb
import shap
import pickle
from explanationplots import *
from sklearn.neighbors import KNeighborsClassifier


class ExplainableModel:

    def __init__(self, x, y, params,
                 feature_names=None,
                 n_neighbors=5,
                 path="./weights/default.pkl",
                 explainer_path="./weights/explainer.pkl",
                 helper_data_path="./weights/helper_data.csv",
                 knn_path="./weights/knn.csv",
                 seed=42):
        self.x = x
        self.y = y
        self.params = params
        self.seed = seed
        self.path = path
        self.explainer_path = explainer_path
        self.helper_data_path = helper_data_path
        self.knn_path = knn_path
        self.feature_names = feature_names
        self.n_neighbor = n_neighbors
        self.model = None
        self.knn = None
        self.explainer = None

    def train(self, iterations=300, early_stopping_rounds=10, do_train=True, load=False, save=True):
        if load:
            self.model = pickle.load(open(self.path, 'rb'))
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
            self.knn = pickle.load(open(self.knn_path, 'rb'))
        if do_train:
            self.model = xgb.XGBClassifier()
            self.model.fit(self.x, self.y)
            self.explainer = shap.TreeExplainer(self.model)
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbor)
            self.knn.fit(self.x, self.y)
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
            with open(self.knn_path, 'wb') as f:
                pickle.dump(self.knn, f)

    def predict(self, x):
        return [round(value) for value in self.predict_proba(x)]

    def predict_proba(self, x):
        if self.model is None:
            self.model = pickle.load(open(self.path, 'rb'))
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
            self.knn = pickle.load(open(self.knn_path, 'rb'))
        return self.model.predict(x)

    def load_explainer(self):
        if self.explainer is None:
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
    def load_knn(self):
        if self.knn is None:
            self.knn = pickle.load(open(self.knn_path, 'rb'))

