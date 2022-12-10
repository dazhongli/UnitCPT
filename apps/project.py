from app import app
import dash_bootstrap_components as dbc
from dash import html
import src.dash_utilities as dash_utl

# input_proj_name = dbc.InputGroup(
#     [dbc.InputGroupText("Project Name"), dbc.Input(
#         placeholder="Project Name")],
#     className="mb-3",
# )

project_form = html.Div([
    dbc.InputGroup(
        [dbc.InputGroupText("Project Name"), dbc.Input(
            placeholder="Project Name")],
        className="mb-3",
    ),
    dbc.Row(
        [
            dbc.Col([

                dbc.InputGroup(
                    [
                        dbc.InputGroupText('Job Number'),
                        dbc.Input(placeholder="Job Number"),
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
            dbc.InputGroupText("Total:"),
            dbc.InputGroupText("$"),
            dbc.Input(placeholder="Amount", type="number"),
            dbc.InputGroupText(".00"),
            dbc.InputGroupText("only"),
        ],
        className="mb-3",
    ),
    dbc.InputGroup(
        [
            dbc.InputGroupText("With textarea"),
            dbc.Textarea(),
        ],
        className="mb-3",
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

btn_new_project = dbc.Button('New Project', id='btn-new-project')


layout = html.Div([
    dbc.Row(
        [
            dbc.Col(project_form),
            dbc.Col('Test')
        ]
    )
])
