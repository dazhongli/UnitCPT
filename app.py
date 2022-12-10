import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State

from src.ags import AGSParser
import src.geoplot as plt
import src.utilities as ult
from src.cpt import CPT

app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.SPACELAB])
server = app.server
app.config.suppress_callback_exceptions = True
app.title = 'Offshore Cone Penetration Test and Foundation Design'
