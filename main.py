from explanationplots import *
from model import ExplainableModel
from sklearn.model_selection import train_test_split
from explanationtexts import *
from dashtabs import *
from dash import Dash, dcc, html, Input, Output
import os
from flask_caching import Cache
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
app = Dash(__name__)

timeout = 20


# all dash callbacks
@app.callback(
    Output('single-feature-plot', 'figure'),
    Input('dropdown-feature', 'value')
)
def change_feat_view(value):
    index = feat_names.index(value)
    return create_detailed_feature_plot(model, current_x, index, value, x_max=100000)


"""
@app.callback(
    Output('prediction', 'children'),
    Output('probability', 'children'),
    Input('transaction', 'value'),
    Input('old-konto-orig', 'value'),
    Input('new-konto-orig', 'value'),
    Input('old-konto-dest', 'value'),
    Input('new-konto-dest', 'value')
)
def change_prediction(transaction, old_konto_orig, new_konto_orig, old_konto_dest, new_konto_dest):
    x_new = [[transaction, old_konto_orig, new_konto_orig, old_konto_dest, new_konto_dest]]
    df = pd.DataFrame(x_new, columns=feat_names)
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][prediction]

    predic_text = "Vorhersage"
    predic_text += "Kein Betrug" if prediction == 0 else "Betrug"
    prob_text = "Wahrscheinlichkeit: " + str(prob)
    return predic_text, prob_text
"""

current_x = None


def get_data():
    df = pd.read_csv("data/paysim.csv")
    df = df.drop("isFlaggedFraud", axis=1)

    df = df.drop(['nameOrig', 'nameDest', 'type', 'step'], axis=1)

    df = df.reset_index(drop=True).rename(columns=feature_names)

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

    # current workaround to deal with the callbacks
    current_x = df_fraud.iloc[0]

    table_basic_1, table_basic_2, fig_feat_basic, fig_class_basic = \
        create_introduction_page_fig(model, df_fraud.iloc[0], feat_names, show_feature_amount=3)
    neighbors_tables = create_deep_dive_page_fig(model, df_fraud.iloc[0], feat_names)

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
                    html.H3(f"Klassifizierung: {label}          Wahrscheinlichkeit: {probability:,.2f}"),
                    html.Table(
                        [html.Tr([html.Th(
                            dcc.Graph(
                                id='basic-table-1',
                                figure=table_basic_1
                            ),
                            rowSpan=2
                        ),
                            html.Th(
                                dcc.Graph(
                                    id='class-plot',
                                    figure=fig_class_basic
                                )
                            ),
                            html.Th(
                                dcc.Graph(
                                    id='basic-table-2',
                                    figure=table_basic_2
                                ),
                                rowSpan=2
                            )
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
                    dcc.Dropdown(feat_names, feat_names[0], id='dropdown-feature'),
                    dcc.Graph(id='single-feature-plot')
                ]),
                html.Div([html.H2("Regelbasierte Textgenerierung")] +
                         [html.P(t) for t in text]
                         )
            ]),
            dcc.Tab(label='Ähnliche Fälle', value='tab-3', children=[
                html.Div([
                    html.H2("Allgemeine Übersicht"),
                    html.H3(f"Klassifizierung: {label}          Wahrscheinlichkeit: {probability:,.2f}"),
                    html.Table(
                        [html.Tr([html.Th(
                            dcc.Graph(
                                id=f'compare-table-{i}',
                                figure=t
                            )
                        ) for i, t in enumerate(neighbors_tables)])
                        ]
                    )
                ])
            ])

        ])
    ])
    """,
                dcc.Tab(label='Interaktiver Tab', value='tab-4', children=[
                    html.Div([
                        html.H2("Ändere Parameter um die Vorhersage zu ändern"),
                        html.P("Transaktionswert:"),
                        dcc.Slider(id='transaction', min=0, max=1000000, step=50000, value=current_x[0],
                                   marks={x: str(x) for x in range(0, 1000000, 100000)}),
                        html.P("alter Kontostand Sender:"),
                        dcc.Slider(id='old-konto-orig', min=0, max=1000000, step=50000, value=current_x[1],
                                   marks={x: str(x) for x in range(0, 1000000, 100000)}),
                        html.P("neuer Kontostand Sender:"),
                        dcc.Slider(id='new-konto-orig', min=0, max=1000000, step=50000, value=current_x[2],
                                   marks={x: str(x) for x in range(0, 1000000, 100000)}),
                        html.P("alter Kontostand Empfänger:"),
                        dcc.Slider(id='old-konto-dest', min=0, max=1000000, step=50000, value=current_x[3],
                                   marks={x: str(x) for x in range(0, 1000000, 100000)}),
                        html.P("neuer Kontostand Empfänger:"),
                        dcc.Slider(id='new-konto-dest', min=0, max=1000000, step=50000, value=current_x[4],
                                   marks={x: str(x) for x in range(0, 1000000, 1000000)}),
                    ]),
                    html.Div([
                        html.H2("Veränderter Wert"),
                        html.P(id='prediction'),
                        html.P(id='probability')
                    ])
                ])"""
    app.run_server(debug=True)
