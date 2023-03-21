import os
from pathlib import Path

import dash_bootstrap_components as dbc
import geopandas as gpd
import pandas as pd
import plotly.express as px
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State

from app import PROJ_DATA, PROJECT_PATH, app
from UnitCPT.src.cpt import CPT
from UnitCPT.src.dash_plot import DashPlot
import UnitCPT.src.geoplot as plt
# ========================================[Global Variables]========================================
px.set_mapbox_access_token(open('./data/mapbox/mapbox_token').read())
cpt_driver = CPT(net_area_ratio=0.85)
# cpt_driver.read_ASCII()
__project_name__ = PROJ_DATA['active_project']

__dash_plotter__ = DashPlot(PROJ_DATA)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 60,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "border": "solid"
}

CONTENT_STYLE = {
    "position": "fixed",
    "top": 60,
    "left": "22rem",
    "bottom": 0,
    "width": "100rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "border": "solid"
}
# ========================================[Global Functions]========================================

# ========================================[Components]========================================

modal_excel = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle('Excel')),
        dbc.ModalBody(
            dbc.Form(
                dbc.Row(
                    [
                        dbc.Label('Select x value', width='auto'),
                        dcc.Dropdown(
                            options=[1, 2, 3],
                            id='dropdown-excel-x'
                        ),
                        dbc.Label('Select y value', width='auto'),
                        dcc.Dropdown(
                            options=[1, 2, 3],
                            id='dropdown-excel-y'
                        ),
                        dbc.Button('Submit', id='btn-modal-submit-excel')
                    ]
                )
            )
        )
    ],
    id='modal-excel',
    is_open=False
)

# -----------------------------------------[CPT Control]---------------------------------------------
cpt_control = dbc.Card(
    [
        dbc.Row(html.H5('Input')),
        dbc.Row(                                        # Area Radio
            [
                dbc.Col(dbc.Label('Area Ratio'), width=5),
                dbc.Col(dbc.Input('input-area-ratio', value=0.85), width=6)
            ]
        ),
        dbc.Row(                                        # Area Radio
            [
                dbc.Col(dbc.Label('Read ASCII'), width=5),
                dbc.Col(dcc.Dropdown(id='dropdown-ASCII',
                                     options=[{'label': 'No file selected',
                                               'value': 'No file selected'}],
                                     value='No file selected'), width=6)
            ]
        ),
        dbc.Row(                                        # JSON
            [
                dbc.Col(dbc.Label('Read JSON'), width=5),
                dbc.Col(dcc.Dropdown(id='dropdown-JSON',
                                     options=[{'label': 'No file selected',
                                               'value': 'No file selected'}],
                                     value='No file selected'), width=6)
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label('CPT Process'), width=5),
                dbc.Col(dbc.Button('Site Boundary'), width=6),
                dbc.Col(dbc.Checkbox(id='cbx-site-boundary'), width=1)
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label('Show Layout'), width=5),
                dbc.Col(dbc.Button('Layout', id='btn-show-layout'), width=6)
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Label('upload Excel'), width=5),
                dbc.Col(dbc.Button('Upload', id='btn-upload-excel')),
                modal_excel
            ]
        )
    ],
    # style=SIDEBAR_STYLE
)

# -----------------------------------------[CPT-Output]-------------------------------------
content_DIV = dbc.Card(
    [
        dbc.Row(
            html.H5('Location Plan')
        ),
        dcc.Graph(id='graph-location-plan', figure={}),
    ],
    # style=CONTENT_STYLE
)
cpt_plot = dbc.Card(
    [
        dbc.Row(
            html.H5('CPT Plot')
        ),
        dcc.Graph(id='fig-cpt-plot', figure={}),
    ],
    # style=CONTENT_STYLE
)
# ========================================[Layout]========================================


def layout():
    layout = dbc.Row(
        [
            dbc.Col(cpt_control, width=4),
            dbc.Col([
                dbc.Row([content_DIV]),
                dbc.Row([cpt_plot])
            ],width=8)
        ], justify='around')
    return layout

# ========================================[Callbacks]========================================


@ app.callback(
    Output('graph-location-plan', 'figure'),
    Input('btn-show-layout', 'n_clicks')
)
def show_CPT_location(n_clicks):
    '''
    Shows the location of CPTs, when user click the show layout button, this will gives a location plan
    '''
    project_name = PROJ_DATA['active_project']
    filename = PROJECT_PATH / project_name / 'data' / 'shp' / 'CPT_coords.json'
    assert (filename.exists())
    gdf = gpd.read_file(filename)
    fig = plt.GEOPlot.get_figure(orientation='h')
    # fig = __dash_plotter__.plot_scatter(gdf)
    fig = __dash_plotter__.plot_point_mapbox(
        gdf=gdf, hoverinfo='CPT', fig=fig, instrument_type='CPT', mode='markers+text')
    # fig = px.scatter_mapbox(gdf, lat='Lon', lon='Lat',
    #                         height=500, width=900, zoom=11)
    fig.update_layout(width=1200, margin=dict(l=0, r=0, t=0, b=0))
    return fig


@ app.callback(
    Output('modal-excel', 'is_open'),
    Input('btn-upload-excel', 'n_clicks'),
    Input('btn-modal-submit-excel', 'n_clicks'),
    State('modal-excel', 'is_open')
)
def toggle_modal(open_modal_click, submit_click, is_open):
    if open_modal_click or submit_click:
        return not is_open


@app.callback(
    [Output('fig-cpt-plot', 'figure'),
     Output('dropdown-ASCII', 'options'),
     Output('dropdown-ASCII', 'value')],
    [Input('graph-location-plan', 'clickData')],
    prevent_initial_callbacks=True
)
def show_CPT(clickData):
    try:
        if clickData is None or 'points' not in clickData:
            return [{}, {}, '']

        point = clickData['points'][0]
        SI_name = point['text']

        project_name = PROJ_DATA['active_project']
        ascii_dir = Path(os.path.join('.', 'projects', project_name, 'ASCII'))
        CPT_dir = Path(os.path.join('.', 'projects', project_name, 'CPT'))
        file_pattern = f'*{SI_name}*'
        files_ASCII = list(ascii_dir.glob(file_pattern))
        files_CPT = list(CPT_dir.glob(file_pattern))
        if len(files_CPT) == 0:
            return [None, None, f'No files found for {SI_name}.']
        options = [{"label": x.stem, "value": x.stem} for x in files_ASCII]
        df = pd.read_json(files_CPT[0])
        cpt_plot = CPT()
        cpt_plot.df = df
        fig = cpt_plot.plot_SBTn_full(SI_name)
        return [fig, options, files_ASCII[0].stem]

    except Exception as e:
        return [{}, {}, f'Error: {e}']
