import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from explanationneighbors import *
from utils import dummy_to_label


def create_feature_importance_plot(model, x_pred, feature_names, show_feature_amount=5):
    explain_val = x_pred.to_numpy().reshape(1, -1)

    model.load_explainer()
    shap = model.explainer.shap_values(explain_val)[0]

    base = model.explainer.expected_value

    val = sum(shap)
    if val < base:
        shap = -shap

    # remove index from shap value
    shap = shap
    mean_importance = np.abs(shap)
    # sort by feature importance
    feature_importance = pd.DataFrame(list(zip(feature_names, mean_importance, shap)),
                                      columns=['col_name', 'feature_importance_val', 'shap_value'])
    feature_importance.sort_values(by=['feature_importance_val'], ascending=False, inplace=True)

    values = []
    for i in range(show_feature_amount):
        feat = feature_importance.iloc[i]
        values.append([feat['col_name'], feat["shap_value"]])

    sum_rest = feature_importance.iloc[show_feature_amount:]['shap_value'].sum(axis=0)
    values.append(["andere", sum_rest])
    values.reverse()

    df = pd.DataFrame(values, columns=['label', 'value'])
    df["color"] = np.where(df["value"] < 0, 'red', 'blue')
    max_val = feature_importance.iloc[0]["shap_value"]

    fig = px.bar(y=df.index, x=df.value, color=df["color"], orientation='h', text=df.label)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition='inside')
    fig.update_layout(xaxis=dict(range=[-max_val, max_val]))

    return fig


def create_class_cluster(model, x_pred):
    x = model.x
    y = np.array(["Betrug" if f == 1 else "Kein Betrug" for f in model.y])

    scaler = StandardScaler().fit(x)
    scaled_x = scaler.transform(x)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_x)

    pred_comp = pca.transform(scaler.transform([x_pred]))

    x_comp = components[:, 0]
    y_comp = components[:, 1]

    fig = px.scatter(x=[pred_comp[0][0]], y=[pred_comp[0][1]]).update_traces(marker_size=20, marker_color="yellow")
    fig.add_traces(
        px.scatter(x=x_comp, y=y_comp, color=y, width=800, height=400).data
    )
    return fig


def create_detailed_feature_plot(model, x_pred, index, feature, x_min=0, x_max=100000):
    # my guess: this works better for continuous features than class and classifier/enum features...
    x = model.x
    y = model.y

    df = x
    df["class"] = y
    val = x_pred[index]
    min_v = x_min if x_min < val else val
    max_v = x_max if x_max > val else val
    fraud = df[(df["class"] == 1) & (df[feature] > min_v) & (df[feature] < max_v)][feature]
    normal = df[(df["class"] == 0) & (df[feature] > min_v) & (df[feature] < max_v)][feature]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=fraud,
        histnorm='probability density',
        name='Betrug'
    ))
    fig.add_trace(go.Histogram(
        x=normal,
        histnorm='probability density',
        name='Kein Betrug'
    ))


    fig.add_vline(x=val, line_width=3, line_color="green")
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    # fig.update_layout(xaxis=dict(range=[x_min, x_max]))

    return fig

def create_type_plot(model, x_pred):
    # my guess: this works better for continuous features than class and classifier/enum features...
    x = model.x
    y = model.y

    df = x
    df["class"] = y
    val = dummy_to_label(x_pred)

    fraud = df[(df["class"] == 1)]
    normal = df[(df["class"] == 0)]
    fraud_list = fraud.apply(dummy_to_label, axis=1)
    normal_list = normal.apply(dummy_to_label, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=fraud_list,
        histnorm='probability density',
        name='fraud'
    ))
    fig.add_trace(go.Histogram(
        x=normal_list,
        histnorm='probability density',
        name='normal'
    ))


    fig.add_vline(x=val, line_width=3, line_color="green")
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    # fig.update_layout(xaxis=dict(range=[x_min, x_max]))

    return fig


def create_table(x, feature_names):
    values = dict(values=[feature_names, x],
                  fill_color='lavender',
                  align='left')
    fig = go.Figure(data=[go.Table(header=dict(values=['Features', 'Values']),
                                    cells=values)])
    fig.update_layout(width=500, height=500)
    return fig

