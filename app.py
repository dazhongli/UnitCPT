import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import yaml
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State

import src
import src.geoplot as plt
import src.utilities as ult
from src.ags import AGSParser
from src.cpt import CPT

current_path = os.path.abspath(__file__)
# add package to path
sys.path.append(os.path.dirname(os.path.dirname(current_path)))

app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.SPACELAB])
server = app.server
app.config.suppress_callback_exceptions = True
app.title = 'Offshore Cone Penetration Test and Foundation Design'
# Global Parameters
PROJECT_PATH = Path('./projects')  # default path for the project

PROJ_DATA = ult.read_input_file('./data.yml')
PROJ_DATA['mapbox_token'] = open('./data/mapbox/mapbox_token').read()
PROJ_DATA['proj_path'] = PROJECT_PATH


def save_project_data():  # save the current data to a file for the next load
    with open('data.yml', 'w') as outfile:
        yaml.dump(PROJ_DATA, outfile, default_flow_style=False)


def get_project_data():  # read from the config files
    PROJ_DATA = ult.read_input_file('./data.yml')
    return PROJ_DATA
