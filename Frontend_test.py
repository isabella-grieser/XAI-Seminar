import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go


# Create the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("XAI for Fraud Detection"),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Allgemeine Übersicht', value='tab-1', children=[

        ]),
        dcc.Tab(label='Deep Dive Verdachtsfall', value='tab-2', children=[
            html.Div([
                html.H2("Verteilungen der Featurewerte für die einzelnen Features + wo der jeweilige Datenpunkt liegt"),
                dcc.Graph(
                    #id='feature-importance-plot',
                    #figure=create_fig1()
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)