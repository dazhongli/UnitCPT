from pathlib import Path
import os
import base64
import logging


import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output, State

from app import app, PROJ_DATA, PROJECT_PATH

# Define the logger below
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s -%(pathname)s:%(lineno)d %(levelname)s - %(message)s', '%y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
# -----------------------------------------------------------------------------------------------------

px.set_mapbox_access_token(open('./data/mapbox/mapbox_token').read())

header = dbc.Row(dcc.Markdown(
    '''
    Data Processing of the Earthquake records
    '''
))

data_input_row = dbc.Row(
    [
        dbc.Col(html.Div(
            id='data-view')),
        dbc.Col(
            [
                dbc.Row(
                    [
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div(['Drag and Drop or',
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
                            multiple=True
                        ),
                        html.Div(id='Output_data_upload')]),
                dbc.Row(
                    [
                        dcc.Dropdown(
                            [i.name for i in Path('./Data').glob('*')], id='file-dropdown'
                        ),
                        dbc.Col([dbc.Button(
                            'Show Data', id='btn-show-data')]),
                        dbc.Spinner(html.Div(id="loading-output"))]
                )
            ]
        )
    ]
)

# def parse_content(contents, filename, date):
#     content_type, content_string = contents.split(',')

#     decoded = base64.b64decodeS(content_string)
#     try:
#         if 'csv' in filename:
#             df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#     except Exception as e:


def layout():
    return [header, data_input_row]

# Functions


def save_file(UPLOAD_DIRECTORY: Path, name, content):
    if not UPLOAD_DIRECTORY.exists():
        UPLOAD_DIRECTORY.mkdir()
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(UPLOAD_DIRECTORY / name, "wb") as fp:
        fp.write(base64.decodebytes(data))
        logger.debug(f'{name} updated to {UPLOAD_DIRECTORY}')


@ app.callback(Output('Output_data_upload', 'children'),
               Input('upload-data', 'contents'),
               State('upload-data', 'filename'),
               State('upload-data', 'last_modified')
               )
def update_output(contents, filenames, last_modified):
    upload_path = PROJECT_PATH / PROJ_DATA['active_project'] / 'ags'
    for filename, content in zip(filenames, contents):
        save_file(upload_path, filename, content)
    return html.Div(f'{filename}')


@ app.callback(
    Output('data-view', 'children'),
    Output('loading-output', 'children'),
    Input('btn-show-data', 'n_clicks'),
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
