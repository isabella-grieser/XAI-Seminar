import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from explanationneighbors import *


def create_feature_importance_plot(model, x_pred, feature_names, show_feature_amount=5):
    explain_val = x_pred.to_numpy().reshape(1, -1)

    model.load_explainer()
    shap = model.explainer.shap_values(explain_val)[0]

    base = model.explainer.expected_value

    val = sum(shap)
    if val > base:
        shap = -shap

    # remove index from shap value
    shap = shap[1:]
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
    values.append(["other", sum_rest])
    values.reverse()

    df = pd.DataFrame(values, columns=['label', 'value'])
    df["positive"] = df["value"] > 0

    max_val = feature_importance.iloc[0]["shap_value"]

    fig = px.bar(y=df.index, x=df.value, color=df.positive, orientation='h', text=df.label)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition='inside')
    fig.update_layout(xaxis=dict(range=[-max_val, max_val]))

    return fig


def create_class_cluster(model, x_pred):
    x = model.x
    y = np.array(["Fraud" if f == 1 else "Not Fraud" for f in model.y])

    scaler = StandardScaler().fit(x)
    scaled_x = scaler.transform(x)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_x)

    x_comp = components[:, 0]
    y_comp = components[:, 1]

    total_var = pca.explained_variance_ratio_.sum() * 100
    fig = px.scatter(x=x_comp, y=y_comp, color=y, title=f'Total Explained Variance: {total_var:.2f}%')

    pred_comp = pca.transform(scaler.transform([x_pred]))
    fig.add_scatter(x=[pred_comp[0][0]], y=[pred_comp[0][1]], mode='markers', marker=dict(size=15, color='green'),
                    showlegend=False)

    return fig


def create_detailed_feature_plot(model, x_pred, index, feature, x_min=0, x_max=100000):
    # my guess: this works better for continuous features than class and classifier/enum features...
    x = model.x
    y = model.y

    df = x
    df["class"] = y

    fraud = df[(df["class"] == 1) & (df[feature] > x_min) & (df[feature] < x_max)][feature]
    normal = df[(df["class"] == 0) & (df[feature] > x_min) & (df[feature] < x_max)][feature]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=fraud,
        histnorm='probability density',
        name='fraud'
    ))
    fig.add_trace(go.Histogram(
        x=normal,
        histnorm='probability density',
        name='normal'
    ))

    fig.add_vline(x=x_pred[index], line_width=3, line_color="green")
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    # fig.update_layout(xaxis=dict(range=[x_min, x_max]))

    return fig


def create_table(x, feature_names):
    values = dict(values=[feature_names, x],
                  fill_color='lavender',
                  align='left')
    return go.Figure(data=[go.Table(header=dict(values=['Features', 'Values']),
                                    cells=values)])


def create_introduction_page_fig(model, x_pred, feature_names, show_feature_amount=3):
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{}],
               [{}]]
    )
    list_val = x_pred.to_numpy().reshape(1, -1)[0][1:].tolist()
    table1 = create_table(list_val, feature_names)
    neighbors, _ = get_n_neighbors_information(model, x_pred, n_neighbors=1)
    table2 = create_table(neighbors, feature_names)
    feat_fig = create_feature_importance_plot(model, x_pred, feature_names, show_feature_amount=show_feature_amount)
    class_fig = create_class_cluster(model, x_pred)

    return table1, table2, feat_fig, class_fig

def create_deep_dive_page_fig(model, x_pred, feature_names):
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{}],
               [{}]]
    )

    neighbors, _ = get_n_neighbors_information(model, x_pred, n_neighbors=4)
    table1 = create_table(neighbors, feature_names)
    table2 = create_table(neighbors, feature_names)
    table3 = create_table(neighbors, feature_names)
    table4 = create_table(neighbors, feature_names)

    return table1, table2, table3, table4
