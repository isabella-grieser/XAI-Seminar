import base64

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from explanationcounterfactuals import get_n_counterfactuals
from explanationplots import *
from model import ExplainableModel
from sklearn.model_selection import train_test_split
from explanationtexts import *
from dashtabs import *
from dash import Dash, dcc, html, Input, Output
from utils import *

feature_description = {
    "Transaktionswert": "Der Wert dieser Transaktion",
    "alter Kontostand Sender": "Der alte Kontostand des Senders",
    "neuer Kontostand Sender": "Der neue Kontostand des Senders",
    "alter Kontostand Empfänger": "Der alte Kontostand des Empfängers",
    "neuer Kontostand Empfänger": "Der neue Kontostand des Empfängers",
    "Zahlung": "Die Zahlungsart: Zahlung",
    "Transfer": "Die Zahlungsart: Transfer",
    "Auszahlung": "Die Zahlungsart: Auszahlung",
    "Debit": "Die Zahlungsart: Debit",
    "Einzahlung": "Die Zahlungsart: Einzahlung"
}

feature_names = {
    "isFraud": "isFraud",
    "amount": "Transaktionswert",
    "oldbalanceOrg": "alter Kontostand Sender",
    "newbalanceOrig": "neuer Kontostand Sender",
    "oldbalanceDest": "alter Kontostand Empfänger",
    "newbalanceDest": "neuer Kontostand Empfänger",
    "type_PAYMENT": "Zahlung",
    "type_TRANSFER": "Transfer",
    "type_CASH_OUT": "Auszahlung",
    "type_DEBIT": "Debit",
    "type_CASH_IN": "Einzahlung",
}
continuous_variables = [feature_names["amount"], feature_names["oldbalanceOrg"], feature_names["newbalanceOrig"],
                        feature_names["oldbalanceDest"], feature_names["newbalanceDest"]]

# Create the Dash app
app = Dash(__name__)

timeout = 20

current_x = None
columns = None
detailed_plots = {}
# all dash callbacks
@app.callback(
    Output('single-feature-plot', 'figure'),
    Input('dropdown-feature', 'value')
)
def change_feat_view(value):
    return detailed_plots[value]


@app.callback(
    Output('prediction', 'children'),
    Output('probability', 'children'),
    Input('transaction', 'value'),
    Input('old-konto-orig', 'value'),
    Input('new-konto-orig', 'value'),
    Input('old-konto-dest', 'value'),
    Input('new-konto-dest', 'value'),
    Input('type-feature', 'value')
)
def change_prediction(transaction, old_konto_orig, new_konto_orig, old_konto_dest, new_konto_dest, type):
    x_new = [transaction, old_konto_orig, new_konto_orig, old_konto_dest, new_konto_dest]
    x_new = label_to_dummy(x_new, type)
    df = pd.DataFrame([x_new], columns=columns)
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][prediction]

    predic_text = "Vorhersage: "
    predic_text += "Kein Betrug" if prediction == 0 else "Betrug"
    prob_text = "Wahrscheinlichkeit: " + str(prob)
    return predic_text, prob_text

def get_data():
    df = pd.read_csv("data/paysim.csv")
    df = df.drop("isFlaggedFraud", axis=1)

    df = df.drop(['nameOrig', 'nameDest', 'step'], axis=1)
    df = pd.get_dummies(df, columns=['type'])
    #there are massive performance issues because the dataset is far too big
    # possible solution: drop random values over oversampled class isFraud=0
    # remove_n = 3000000
    # drop_indices = np.random.choice(df[df['isFraud'] == 0].index, remove_n, replace=False)
    # df = df.drop(drop_indices)
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

    df_test = x_test
    df_test["label"] = y_hat
    df_fraud = df_test[df_test["label"] == 1]
    df_fraud = df_fraud.drop(['label'], axis=1)

    print(y_hat)
    print("F1: " + str(f1_score(y_test, y_hat)))
    print("Precision: " + str(precision_score(y_test, y_hat)))
    print("Recall: " + str(recall_score(y_test, y_hat)))
    print("Accuracy: " + str(accuracy_score(y_test, y_hat)))

    # current workaround to deal with the callbacks
    current_x = df_fraud.iloc[0]
    columns = df_fraud.columns.tolist()

    table_basic_1, table_basic_2, fig_feat_basic, fig_class_basic = \
        create_introduction_page_fig(model, df_fraud.iloc[0], df_fraud.columns, show_feature_amount=3)
    neighbors_tables = create_neighbors_page_fig(model, df_fraud[0:1], df_fraud.columns)

    text = create_explanation_texts(model, df_fraud[0:1], 1, df_fraud.columns, feature_description)

    for index, f in enumerate(df_fraud.columns.tolist()[:-5] + ["Transaktionsart"]):
        if f != "Transaktionsart":
            detailed_plots[f] = create_detailed_feature_plot(model, current_x, index, f, x_max=100000)
        else:
            detailed_plots[f] = create_type_plot(model, current_x)

    label = "Betrug"
    probability = model.predict_proba(df_fraud[0:1])[0][1]

    ing_png = 'ING_new.png'
    ing_base64 = base64.b64encode(open(ing_png, 'rb').read()).decode('ascii')

    # define the layout
    app.layout = html.Div([
        html.Img(src='data:image/png;base64,{}'.format(ing_base64),
                 style={
                     'float': 'right',
                     "width": "100px",
                     "height": "auto"
    }),
        html.H1("XAI für Betrugserkennung"),
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='Allgemeine Übersicht', value='tab-1', children=[
                html.Div([
                    html.H2("Allgemeine Übersicht"),
                    html.H3(f"Klassifizierung: {label}"),
                    html.H3(f"Wahrscheinlichkeit: {probability:,.2f}"),
                    html.Table(
                        [html.Tr([html.Th(
                            dcc.Graph(
                                id='basic-table-1',
                                figure=table_basic_1
                            ),
                            style={'vertical-align': 'baseline'}
                        ),
                            html.Th(html.Div(className='plots', children=[
                                dcc.Graph(
                                    id='basic-feat-view',
                                    figure=fig_feat_basic,
                                    #style={'position': 'relative', 'top': '-100px'}
                                ),
                                dcc.Graph(
                                    id='class-plot',
                                    figure=fig_class_basic)
                            ])),
                            html.Th(
                                dcc.Graph(
                                    id='basic-table-2',
                                    figure=table_basic_2,
                                    #style={'position': 'relative', 'top': '-150px'}
                                ),
                                style={'vertical-align': 'baseline'}
                            )
                        ])
                        ]
                    )
                ]),
                html.Div([html.H2("Regelbasierte Textgenerierung")] +
                         [html.P(t) for t in text]
                         )
            ]),
            dcc.Tab(label='Deep Dive Verdachtsfall', value='tab-2', children=[
                html.Div([
                    html.H2(
                        "Verteilungen der Featurewerte für die einzelnen Features"),
                    dcc.Dropdown(df_fraud.columns.tolist()[:-5] + ["Transaktionsart"], df_fraud.columns[0], id='dropdown-feature'),
                    dcc.Graph(id='single-feature-plot')
                ])
            ]),
            dcc.Tab(label='Ähnliche Fälle', value='tab-3', children=[
                html.Div([
                    html.H2("Allgemeine Übersicht"),
                    html.H3(f"Klassifizierung: {label}"),
                    html.H3(f"Wahrscheinlichkeit: {probability:,.2f}"),
                    html.Table(
                        [html.Tr([html.Th(
                            dcc.Graph(
                                id=f'compare-table-{i}',
                                figure=t
                            )
                        ) for i, t in enumerate(neighbors_tables)])
                        ],
                        style={"cellspacing": "0",
                                 "border-collapse": "collapse"
                                 }
                    )
                ])
            ]),
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
                               marks={x: str(x) for x in range(0, 1000000, 100000)}),
                    html.P("Zahlungsart:"),
                    dcc.Dropdown(["Zahlung", "Transfer", "Auszahlung", "Debit", "Einzahlung"]
                                 , dummy_to_label(current_x), id='type-feature'),
                ]),
                html.Div([
                    html.H2("Veränderter Wert"),
                    html.P(id='prediction'),
                    html.P(id='probability')
                ])
            ])
        ])
    ])

    app.run_server(debug=True)
