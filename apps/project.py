from app import app
import dash_bootstrap_components as dbc
from dash import html
import src.dash_utilities as dash_utl
from pathlib import Path
from dash import Input, Output, State, dcc
import logging

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
                        dbc.Input(placeholder="Amount", type="number"),
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
folders = list(PROJECT_PATH.glob('*'))
if (len(folders) == 0):
    projects = html.Div('No Project')
else:
    projects = dbc.Row(
        [
            dbc.Label("Existing Projects",
                      html_for="example-radios-row", width=2),
            dbc.Col(
                dbc.RadioItems(
                    id="radio-proj",
                    options=[{'label': f.stem, 'value': f.stem}
                             for f in folders]
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
            dbc.Col(dbc.Form([projects, btn_new_project]))
        ]
    ),
    dcc.ConfirmDialog(id='cfm-project-created',
                      message='Project has been created')
])


@app.callback(
    Output('cfm-project-created', 'displayed'),
    Output('cfm-project-created', 'message'),
    Output('radio-proj', 'options'),
    State('input-name', 'value'),
    State('input-JN', 'value'),
    Input('btn-create-proj', 'n_clicks'), prevent_initial_call=True
)
def display_confirm(name, JN, n_clicks):
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
