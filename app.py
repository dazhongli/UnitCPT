from dash.dependencies import Input, Output, State
from dash import dash_table, dcc, html
import yaml
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import src
import dash
from pathlib import Path
import sys

import src.geoplot as plt
import src.utilities as ult
from src.ags import AGSParser
from src.cpt import CPT

app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.SPACELAB])
server = app.server
app.config.suppress_callback_exceptions = True
app.title = 'Offshore Cone Penetration Test and Foundation Design'
# Global Parameters
PROJECT_PATH = Path('./projects')  # default path for the project
PROJ_DATA = ult.read_input_file('./data.yml')
PROJ_DATA['mapbox_token'] = open('./data/mapbox/mapbox_token').read()


def save_project_data():  # save the current data to a file for the next load
    with open('data.yml', 'w') as outfile:
        yaml.dump(PROJ_DATA, outfile, default_flow_style=False)


def get_project_data():  # read from the config files
    PROJ_DATA = ult.read_input_file('./data.yml')
    return PROJ_DATA
