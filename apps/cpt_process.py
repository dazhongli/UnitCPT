from src.cpt import CPT
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

# ------------------Global------------------------------------------------
cpt_driver = CPT(net_area_ratio=0.85)
cpt_driver.read_ASCII()
# ------------------Component------------------------------------------------
# ------------------Layout------------------------------------------------
# ------------------Callbacks------------------------------------------------
