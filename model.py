import xgboost as xgb
import shap
import pickle
from explanationplots import *
from sklearn.neighbors import KNeighborsClassifier
import dice_ml

class ExplainableModel:

    def __init__(self, x, y, params,
                 feature_names=None,
                 n_neighbors=5,
                 path="./weights/default.pkl",
                 explainer_path="./weights/explainer.pkl",
                 explainer_dice_path="./weights/explainer_dice.pkl",
                 continuous_variables=None,
                 helper_data_path="./weights/helper_data.csv",
                 knn_path="./weights/knn.pkl",
                 seed=42):
        self.x = x
        self.y = y
        self.params = params
        self.seed = seed
        self.path = path
        self.explainer_path = explainer_path
        self.helper_data_path = helper_data_path
        self.explainer_dice_path = explainer_dice_path
        self.knn_path = knn_path
        self.feature_names = feature_names
        self.n_neighbor = n_neighbors
        self.continuous_variables = continuous_variables
        self.model = None
        self.knn = None
        self.explainer = None
        self.dice = None

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
            d = dice_ml.Data(dataframe=df, continuous_features=self.continuous_variables,
                             outcome_name='isFraud')
            m = dice_ml.Model(model=self.model, backend="sklearn", model_type="classifier")
            self.dice = dice_ml.Dice(d, m, method="random")

            # calculate the medians for the features of the two different classes
            fraud_means = df[df['isFraud'] == 1].median(axis=0)
            normal_means = df[df['isFraud'] == 0].median(axis=0)

            means = pd.concat([fraud_means, normal_means], axis=1).transpose()
            means.to_csv(self.helper_data_path)
        if save:
            with open(self.path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.explainer_path, 'wb') as f:
                pickle.dump(self.explainer, f)
            with open(self.knn_path, 'wb') as f:
                pickle.dump(self.knn, f)
            with open(self.explainer_dice_path, 'wb') as f:
                pickle.dump(self.dice, f)

    def predict(self, x):
        if self.model is None:
            self.model = pickle.load(open(self.path, 'rb'))
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
            self.knn = pickle.load(open(self.knn_path, 'rb'))
            self.dice = pickle.load(open(self.explainer_dice_path, 'rb'))
        return self.model.predict(x)

    def predict_proba(self, x):
        if self.model is None:
            self.model = pickle.load(open(self.path, 'rb'))
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
            self.knn = pickle.load(open(self.knn_path, 'rb'))
            self.dice = pickle.load(open(self.explainer_dice_path, 'rb'))
        return self.model.predict_proba(x)

    def load_explainer(self):
        if self.explainer is None:
            self.explainer = pickle.load(open(self.explainer_path, 'rb'))
            self.dice = pickle.load(open(self.explainer_dice_path, 'rb'))

    def load_knn(self):
        if self.knn is None:
            self.knn = pickle.load(open(self.knn_path, 'rb'))

