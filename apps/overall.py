import base64
import datetime
import io
import os
import subprocess
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State

from app import app

# Define the layout below

px.set_mapbox_access_token(open('./data/mapbox/mapbox_token').read())

header = dbc.Row(dcc.Markdown(
    '''
    Data Processing of the Earthquake records
    '''
))

data_input_row = dbc.Row(
    [
        dbc.Col(html.Div(
            id='earthquake-table')),
        dbc.Col(
            [
                dbc.Row(
                    [
                        dcc.Upload(
                            id='upload_data',
                            children=html.Div([
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Allow multiple files to be uploaded
                            multiple=False
                        ),
                        html.Div(id='Output_data_upload')]),
                dbc.Row(
                    [
                        dcc.Dropdown(
                            [i.name for i in Path('./Data').glob('*')], id='file-dropdown'
                        ),
                        dbc.Col([dbc.Button(
                            'Show Data', id='show-data')]),
                        dbc.Spinner(html.Div(id="loading-output"))]
                )

            ]
        )
    ]
)

layout = [header, data_input_row]


@ app.callback(Output('Output_data_upload', 'children'),
               Input('upload_data', 'filename')
               )
def update_output(filename):
    return html.Div(f'{filename}')


@ app.callback(
    Output('earthquake-table', 'children'),
    Output('loading-output', 'children'),
    Input('show-data', 'n_clicks'),
    State('file-dropdown', 'value'),
    prevent_initial_call=True
)
def show_table(n_clicks, filename):
    if n_clicks is not None:
        file_full_path = Path.cwd() / f'./Data/{filename}'
        print(file_full_path)
        # os.startfile(file_full_path)
        df = pd.read_csv(file_full_path)
        fig = px.scatter_mapbox(df, lat="LAT", lon="LON",
                                height=800, width=1200, color='MAG_FINAL', zoom=3)

        description = dcc.Markdown(f'''

        _Loaded file {filename}_

        |Item|Data|
        |:-----------|:----------|
        |rows        |{df.shape[0]:,.0f}|
        |cols        |{df.shape[1]}|
        ''')
        return [dcc.Graph(figure=fig), description]
