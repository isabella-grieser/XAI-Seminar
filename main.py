from explanationplots import *
from model import ExplainableModel
import pandas as pd
from sklearn.model_selection import train_test_split
from explanationneighbors import *
from explanationcounterfactuals import *
from explanationtexts import *
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go

feature_description = {
    "Transaktionswert": "Der Wert dieser Transaktion",
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

# Create the Dash app
app = dash.Dash(__name__)


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

    # fig3 = create_detailed_feature_plot(model, df_fraud.iloc[0], 0, feature_names["amount"], x_max=200000)
    # fig4 = create_introduction_page_fig(model, df_fraud.iloc[0], feat_names, show_feature_amount=3)
    # neighbors = get_n_neighbors_information(model, df_fraud.iloc[0], n_neighbors=3)
    # counterfactuals = get_n_counterfactuals(model, df_fraud[0:1], n_factuals=4)
    table_basic_1, table_basic_2, fig_feat_basic, fig_class_basic = \
        create_introduction_page_fig(model, df_fraud.iloc[0], feat_names, show_feature_amount=3)

    # fig4.show()
    text = create_explanation_texts(model, df_fraud[0:1], 1, feat_names, feature_description)

    label = "Fraud"
    probability = model.predict_proba(df_fraud[0:1])[0][1]

    # define the layout
    app.layout = html.Div([
        html.H1("XAI for Fraud Detection"),
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='Allgemeine Übersicht', value='tab-1', children=[
                html.Div([
                    html.H2("Allgemeine Übersicht"),
                    html.H3(f"Klassifizierung: {label}    Wahrscheinlichkeit: {probability}"),
                    html.Table(
                        [html.Tr([html.Th(rowSpan=2),
                                  html.Th(
                                      dcc.Graph(
                        id='class-plot',
                        figure=fig_class_basic
                        )),
                         html.Th(rowSpan=2)
                        ]),
                         html.Tr([html.Th(dcc.Graph(
                            id='basic-feat-view',
                            figure=fig_feat_basic)
                            )]
                         )
                        ]
                    )
                ])
            ]),
            dcc.Tab(label='Deep Dive Verdachtsfall', value='tab-2', children=[
                html.Div([
                    html.H2(
                        "Verteilungen der Featurewerte für die einzelnen Features + wo der jeweilige Datenpunkt liegt"),
                    dcc.Graph(
                        id='feature-importance-plot',
                        figure=create_feature_importance_plot(model, df_fraud.iloc[0], feat_names)
                    )
                ]),
                html.Div([
                    html.H2("Evtl. Regelbasierte Texgenerierung"),
                    dcc.Graph(
                        id='deep-dive-plot-2',
                        figure={
                            'data': [
                                go.Bar(x=[1, 2, 3], y=[5, 7, 9], name='Bar')
                            ],
                            'layout': go.Layout(title='Deep Dive Subplot 2')
                        }
                    )
                ])
            ]),
            dcc.Tab(label='Deep Dive Vergleichsfall', value='tab-3', children=[
                html.Div([
                    html.Div([
                        html.H2("Übersicht zum aktuellen Fall (eher top level)"),
                        dcc.Graph(
                            id='comparison-plot-1',
                            figure={
                                'data': [],
                                'layout': go.Layout(title='Comparison Subplot 1')
                            }
                        )
                    ], className='four columns'),
                    html.Div([
                        html.H2("Übersicht zum Vergleichsfall (eher top level)"),
                        dcc.Graph(
                            id='comparison-plot-2',
                            figure={
                                'data': [],
                                'layout': go.Layout(title='Comparison Subplot 2')
                            }
                        )
                    ], className='four columns'),
                    html.Div([
                        html.H2("Übersicht zum Vergleichsfall (eher top level)"),
                        dcc.Graph(
                            id='comparison-plot-3',
                            figure={
                                'data': [],
                                'layout': go.Layout(title='Comparison Subplot 3')
                            }
                        )
                    ], className='four columns')
                ], className='row')
            ]),
            dcc.Tab(label='Interaktiver Tab', value='tab-4', children=[
                html.Div([
                    html.H2("Subplot 7"),
                    dcc.Graph(
                        id='interactive-plot-1',
                        figure={
                            'data': [
                                go.Scatter(x=[1, 2, 3], y=[1, 3, 2], mode='lines', name='Line')
                            ],
                            'layout': go.Layout(title='Interactive Subplot 1')
                        }
                    ),
                    html.H2("Subplot 8"),
                    dcc.Graph(
                        id='interactive-plot-2',
                        figure={
                            'data': [
                                go.Bar(x=[1, 2, 3], y=[2, 4, 6], name='Bar')
                            ],
                            'layout': go.Layout(title='Interactive Subplot 2')
                        }
                    )
                ])
            ])
        ])
    ])

    app.run_server(debug=True)
