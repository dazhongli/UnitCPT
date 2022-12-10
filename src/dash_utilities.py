import dash_bootstrap_components as dbc
from dash import html


def dbc_one_row(component_list):
    html.Div(dbc.Row([dbc.Col([c]) for c in component_list]))


def dbc_one_col(component_list):
    html.Div(dbc.Col([dbc.Row([c]) for c in component_list]))
