from fileinput import filename
import logging
import re
import os
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

import src.dash_utilities as dash_utl
from app import app

# input_proj_name = dbc.InputGroup(
#     [dbc.InputGroupText("Project Name"), dbc.Input(
#         placeholder="Project Name")],
#     className="mb-3",
# )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s -%(pathname)s:%(lineno)d %(levelname)s - %(message)s', '%y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Global
PROJECT_PATH = Path('./projects')  # default path for the project

project_form = html.Div([
    dbc.InputGroup(
        [dbc.InputGroupText("Project Name"),
         dbc.Input(id='input-name',
                   placeholder="Project Name")],
        className="mb-3",
    ),
    dbc.Row(
        [
            dbc.Col([
                dbc.InputGroup(
                    [
                        dbc.InputGroupText('Job Number'),
                        dbc.Input(id='input-JN', placeholder="Job Number"),
                    ],
                    className="mb-5",
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Location"),
                        dbc.Input(id='input-location',
                                  placeholder="Amount", type="number"),
                    ],
                    className="mb-3",
                )
            ])
        ]
    ),
    dbc.InputGroup(
        [
            dbc.Select(
                options=[
                    {"label": "Option 1", "value": 1},
                    {"label": "Option 2", "value": 2},
                ]
            ),
            dbc.InputGroupText("With select"),
        ]
    ),
]
)

# Build the controlling Group


btn_new_project = dbc.Button('New Project', id='btn-create-proj')
btn_delete_project = dbc.Button('Delete Project', id='btn-delete-proj')
btn_open_folder = dbc.Button('Open Folders', id='btn-open-folders')

folders = list(PROJECT_PATH.glob('*'))
if len(folders) == 0:
    radio_options = [{'label': 'No Project', 'value': 'No Project'}]
else:
    radio_options = [{'label': f.stem, 'value': f.stem}
                     for f in folders]
projects = dbc.Row(
    [
        dbc.Label("Existing Projects",
                  html_for="example-radios-row", width=2),
        dbc.Col(
            dbc.RadioItems(
                id="radio-proj",
                options=radio_options
            ),
            width=10,
        ),
    ],
    className="mb-3",
)
layout = html.Div([
    dbc.Row(
        [
            dbc.Col(project_form),
            dbc.Col(dbc.Form([projects,
                             btn_new_project,
                             btn_delete_project,
                             btn_open_folder]))
        ]
    ),
    dcc.ConfirmDialog(id='cfm-project-created',
                      message='Project has been created'),
    dcc.ConfirmDialog(id='cfm-project-deleted',
                      message='Project has been deleted'),
    dbc.Alert(
        'This is a test',
        id='alert-auto',
        is_open=False,
        duration=4000,
    ),
])


@app.callback(
    Output('cfm-project-created', 'displayed'),
    Output('cfm-project-created', 'message'),
    Output('radio-proj', 'options'),
    State('input-name', 'value'),
    State('input-JN', 'value'),
    Input('btn-create-proj', 'n_clicks'), prevent_initial_call=True
)
def create_project(name, JN, n_clicks):
    folders = list(PROJECT_PATH.glob('*'))
    options = [{'label': f.stem, 'value': f.stem} for f in folders]
    if n_clicks != 0:
        if name is None:
            return True, "Please Input File Name", options
        folder_name = f'{JN}-{name[0:min(20,len(name))]}'
        folder_path = PROJECT_PATH / folder_name
        if folder_path.exists():
            return True, f'{folder_path} Already Exist', options
        else:
            folder_path.mkdir()
            folders = list(PROJECT_PATH.glob('*'))
            return True, f'{folder_path} Created', [{'label': f.stem, 'value': f.stem} for f in folders]
    else:
        return False, 'this is also a test'


@app.callback(
    Output('cfm-project-deleted', 'displayed'),
    Output('cfm-project-deleted', 'message'),
    State('radio-proj', 'value'),
    Input('btn-delete-proj', 'n_clicks'), prevent_initial_call=True
)
def delete_project(project_name, n_clicks):
    if n_clicks != 0:
        return True, f'Delete Entire Folder {project_name}?'
# Update the input form while clicking radio button


@app.callback(
    Output('radio_proj', 'options'),
    State('radio-proj', 'value'),
    Input('cfm-project-deleted', 'submit_n_clicks')
)
def delete_folder(proj_name, submit_n_clicks):
    if submit_n_clicks:
        project_folder = PROJECT_PATH / proj_name
        project_folder.unlink()
        folders = list(PROJECT_PATH.glob('*'))
        logger.info(f'{filename} deleted!')
        return [{'label': f.stem, 'value': f.stem} for f in folders]


@app.callback(
    Output('alert-auto', 'value'),
    State('radio-proj', 'value'),
    Input('btn-open-folders', 'n_clicks')
)
def open(folder, n_clicks):
    if n_clicks:
        os.startfile(PROJECT_PATH/folder)


@app.callback(
    Output('input-name', 'value'),
    Output('input-JN', 'value'),
    Input('radio-proj', 'value'), present_initial_call=True
)
def update_input_form(label):
    if label is not None:
        JN = label.split('-')[0]
        name = label.split('-')[1]
    else:
        JN = 'None'
        name = 'None'
    return name, JN
