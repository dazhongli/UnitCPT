from app import app
import os
from dash import html, dcc

with open('./ref/reference.md', 'r', encoding="utf8") as fin:
    str_about = fin.read()


layout = html.Div(dcc.Markdown(str_about))
