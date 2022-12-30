import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State

from app import PROJ_DATA, PROJECT_PATH, app
from src.cpt import CPT

# ========================================[Global Variables]========================================
cpt_driver = CPT(net_area_ratio=0.85)
# cpt_driver.read_ASCII()
__project_name__ = PROJ_DATA['active_project']

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 60,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa"
}
# ========================================[Global Functions]========================================

# ========================================[Components]========================================


# -----------------------------------------[CPT Control]---------------------------------------------
cpt_control = html.Div(
    [
        dbc.Row(html.H4('CPT Process')),
        dbc.Row(html.H5('Input')),
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
                dbc.Col(dbc.Button('btn-show-layout'), width=6)
            ]
        )
    ],
    style=SIDEBAR_STYLE
)


# ========================================[Layout]========================================
def layout():
    layout = html.Div([cpt_control])
    return layout


# ========================================[Callbacks]========================================
