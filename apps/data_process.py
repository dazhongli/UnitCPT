from pathlib import Path
import os
import base64
import logging
from src.ags import AGSParser


import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dcc, html, dash_table
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
px.set_mapbox_access_token(open('./data/mapbox/mapbox_token').read())


# ========================================[Global Variables]========================================
__ags_parser__ = AGSParser(ags_str='', ags_format=2)


# ========================================[Global Funcs]========================================
def __save_file__(UPLOAD_DIRECTORY: Path, name, content):
    '''
    Save the files
    '''
    if not UPLOAD_DIRECTORY.exists():
        UPLOAD_DIRECTORY.mkdir()
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(UPLOAD_DIRECTORY / name, "wb") as fp:
        fp.write(base64.decodebytes(data))
        logger.debug(f'{name} updated to {UPLOAD_DIRECTORY}')


# ========================================Components========================================
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
layout_data_table = html.Div(
    dbc.Row(
        [
            dbc.Col(dash_table.DataTable(id='table-index'), width=1),
            dbc.Col(dash_table.DataTable(id='table-content',
                                         style_table={'height': '500px', 'overflowY': 'auto'}), width=10),
            dbc.Col(id='data-control', width=1)
        ]
    )
)


ags_data_control = html.Div(
    dbc.Row(
        [
            dbc.Button('Download Excel', id='btn-download-excel'),
            dbc.Button('Download Json', id='btn-download-json'),
            dcc.Download(id='download-dataframe-excel'),
            dcc.Download(id='download-dataframe-json'),
        ]
    )
)


def convert_data_table(df: pd.DataFrame):
    data = df.to_dict('records')
    columns = [{'name': i, 'id': i} for i in df.columns]
    return data, columns


# ========================================Layout========================================
def layout():
    return [data_input_row, header, layout_data_table]


# ========================================Callbacks========================================
# ---------------------------------------[Save Data]-------------------------------------------------
@ app.callback(Output('Output_data_upload', 'children'),
               Output('table-index', 'data'),
               Output('table-index', 'columns'),
               Input('upload-data', 'contents'),
               State('upload-data', 'filename'),
               State('upload-data', 'last_modified')
               )
def update_output(contents, filenames: list[Path], last_modified):
    upload_path = PROJECT_PATH / PROJ_DATA['active_project'] / 'ags'
    for filename, content in zip(filenames, contents):
        if filename.endswith('ags'):
            upload_path = PROJECT_PATH / PROJ_DATA['active_project'] / 'ags'
        elif filename.endswith('asc'):
            upload_path = PROJECT_PATH / PROJ_DATA['active_project'] / 'ASCII'
        else:
            upload_path = PROJECT_PATH / PROJ_DATA['active_project'] / 'TXT'
            logger.warning(
                f'{filename} extension not known, saved at TXT folder')
        __save_file__(upload_path, filename, content)
        __ags_parser__.read_ags_file(upload_path/filename)
        df = pd.DataFrame()
        df['key'] = __ags_parser__.key_IDs
        data, columns = convert_data_table(df)
    return html.Div(f'{filename}'), data, columns


# ---------------------------------------[AGS Key Callback]-------------------------------------------------
@app.callback(
    Output('table-content', 'data'),
    Output('table-content', 'columns'),
    Output('data-control', 'children'),
    Input('table-index', 'active_cell'), prevent_initial_call=True
)
def show_content(active_cell):
    key = __ags_parser__.key_IDs[active_cell['row']]
    df = __ags_parser__.get_df_from_key(key)
    df_no_na = df.mask(df == '').dropna(thresh=3, axis=1)
    data, columns = convert_data_table(df_no_na)
    return data, columns, ags_data_control
    # get the key selected


# ---------------------------------------[Download Excel]-------------------------------------------------
@app.callback(
    Output('download-dataframe-excel', 'data'),
    Input("btn-download-excel", 'n_clicks'),
    prevent_initial_call=True
)
def download_excel(n_clicks):
    return dcc.send_data_frame(__ags_parser__.active_df.to_excel, f'{__ags_parser__.active_key}.xlsx',
                               sheet_name=f'{__ags_parser__.active_key}')


# ---------------------------------------[Download JSON]-------------------------------------------------
@app.callback(
    Output('download-dataframe-json', 'data'),
    Input("btn-download-json", 'n_clicks'),
    prevent_initial_call=True
)
def download_excel(n_clicks):
    return dcc.send_data_frame(__ags_parser__.active_df.to_json, f'{__ags_parser__.active_key}.json')
