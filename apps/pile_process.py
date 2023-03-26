import base64
import io

import matplotlib.pyplot as plt

from UnitCPT import (Input, Output, Path, PipePile, app, dash_table, dbc, dcc,
                     html, pd, px, PROJ_DATA)
from UnitCPT.apps.io_unitcpt import get_cpt, read_proj_coords

__pile__ = PipePile(dia=3.5, thickness=0.05, length=70, penetration=60)

pile_dim_card = dbc.Card(
    [
        dbc.CardHeader("Pile Dimension"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Label("Pile Diameter (m)",
                                style={"font-size": "12pt"}), width=6),
                        dbc.Col(dcc.Input(id="input-diameter",
                                type="number", value=__pile__.dia_out), width=6),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Pile Thickness (mm)",
                                style={"font-size": "12pt"}), width=6),
                        dbc.Col(dcc.Input(id="input-thickness",
                                type="number", value=__pile__.t*1000), width=6),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Pile Embedment (m)",
                                style={"font-size": "12pt"}), width=6),
                        dbc.Col(dcc.Input(id="input-embedment",
                                type="number", value=__pile__.penetration), width=6),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Pile Length (m)",
                                style={"font-size": "12pt"}), width=6),
                        dbc.Col(dcc.Input(id="input-length",
                                type="number", value=__pile__.length), width=6),
                    ],
                    align="center",
                ),
                html.Div(id="dt-ratio"),
            ]
        ),
    ]
)
# Loading Input card
loading_card = dbc.Card(
    [
        dbc.CardHeader("Loading Input (Resultant)"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Label("Axial Force (MN)"), width=6),
                        dbc.Col(dcc.Input(id="input-axial",
                                type="number", value=0), width=6),
                    ],
                    align="center",
                    style={"margin-bottom": "10px"}
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Shear Force (MN)"), width=6),
                        dbc.Col(dcc.Input(id="input-shear",
                                type="number", value=0), width=6),
                    ],
                    align="center",
                    style={"margin-bottom": "10px"}
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Bending Moment (MN/m)"), width=6),
                        dbc.Col(dcc.Input(id="input-moment",
                                type="number", value=0), width=6),
                    ],
                    align="center",
                    style={"margin-bottom": "10px"}
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Level (m)"), width=6),
                        dbc.Col(dcc.Input(id="input-level",
                                type="number", value=0), width=6),
                    ],
                    align="center",
                    style={"margin-bottom": "10px"}
                ),
            ]
        ),
    ]
)
pile_soil_card = dbc.Card(
    [
        dbc.CardHeader("Pile Soil Interaction"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Label("Method", style={
                                "font-size": "12pt"}), width=3),
                        dbc.Col(
                            dcc.Dropdown(
                                id="input-method",
                                options=[
                                    {"label": "ISO", "value": "iso"},
                                    {"label": "DNV/API", "value": "dnv_api"},
                                ],
                                value="iso",
                            ),
                            width=9,
                        ),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Interval of plotting",
                                style={"font-size": "12pt"}), width=3),
                        dbc.Col(
                            dcc.Slider(
                                id="input-plot-interval",
                                min=0.5,
                                max=8.0,
                                step=0.5,
                                value=2.0,
                                marks={i: str(i)
                                       for i in [0.5, 2.0, 4.0, 8.0]},
                            ),
                            width=9,
                        ),
                    ],
                    align="center",
                ),
            ]
        ),
    ],
    id="pile-soil-card",
)
pile_resp_card = dbc.Card(
    [
        dbc.CardHeader("Pile Response Calculation"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Label("Solver", style={
                                "font-size": "12pt"}), width=3),
                        dbc.Col(
                            dcc.Dropdown(
                                id="input-solver",
                                options=[
                                    {"label": "ALP", "value": "alp"},
                                    {"label": "Optif", "value": "optif"},
                                    {"label": "Plaxis", "value": "plaxis"},
                                ],
                                value="alp",
                            ),
                            width=9,
                        ),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(dbc.Button("Run", id="btn-run"), width=12),
                    ],
                    align="center",
                ),
            ]
        ),
    ],
    id="pile-resp-card",
)
df = read_proj_coords(PROJ_DATA['proj_path'], PROJ_DATA['active_project'])

CPT_card = dbc.Card(
    [
        dbc.CardHeader("CPT Data"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Label('Select CPT Data', style={
                            'fontSize': '12px'}), width=3),
                        dbc.Col(dcc.Dropdown(
                            id='cpt-dropdown',
                            options=[{'label': x, 'value': x}
                                for x in df['CPT']],
                            value=df.index[0]
                        ), width=9)
                    ],
                    align='center'
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label('Upload CPT Data', style={
                            'fontSize': '12px'}), width=3),
                        dbc.Col(dcc.Upload(
                            id='cpt-upload',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ), width=9)
                    ],
                    align='center'
                ),
                html.Div(id='output-data-upload')
            ]
        )
    ]
)

graphs = html.Div(
    [
        html.H4("Output"),
        html.Div(id='fig-pile-shape'),
        dcc.Graph(id="figure-CPT-location"),
        dcc.Graph(id="figure-pile-response"),
    ],
)

# layout = html.Div([sidebar, graphs])

sidebar = html.Div(
    [
        CPT_card,
        pile_dim_card,
        loading_card,
        pile_soil_card,
        pile_resp_card
    ],
    className="sidebar",
)


def layout():
    layout = dbc.Row(
        [
            dbc.Col(sidebar, width=4),
            dbc.Col(graphs, width=8)
        ], justify='left')
    return layout


def resize_figure(fig, width_px, height_px, dpi):
    # Set the figure size in inches based on the desired pixel size and DPI
    width_in = width_px / dpi
    height_in = height_px / dpi
    fig.set_size_inches(width_in, height_in)


# ---------------------------------------[Callback]-------------------------------------------------
# callback for pile dimension


# Pile Shape Update
@app.callback(
    Output('fig-pile-shape', 'children'),
    [Input('input-diameter', 'value'),
     Input('input-thickness', 'value'),
     Input('input-length', 'value'),
     Input('input-embedment', 'value'),
     Input('cpt-dropdown', 'value')
     ],
    prevent_initial_callbacks=True
)
def update_pile_geometry(diameter, thickness, length, penetration, cpt_id):
    __pile__.dia_out = diameter
    __pile__.t = thickness/1000
    __pile__.penetration = penetration
    __pile__.length = length
    __pile__.refresh()

    cpt_filename = f'{cpt_id}.json'
    try:
        cpt = get_cpt(cpt_filename, PROJ_DATA)
        fig, ax = __pile__.plot_cpt(cpt)
    except Exception as e:
        print(e)
        fig, ax = __pile__.plot()
    resize_figure(fig, 4000, 5000, 600)

    # Serialize the plot to a base64-encoded string
    buf = io.BytesIO()
    fig.savefig(buf, format='svg')
    buf.seek(0)
    fig.tight_layout()
    fig.patch.set_edgecolor('blue')
    fig.patch.set_linewidth('2')
    b64 = base64.b64encode(buf.read()).decode('utf-8')

    # Return the serialized plot as an image component
    return html.Div(
        html.ObjectEl(
            data='data:image/svg+xml;base64,{}'.format(b64),
            type='image/svg+xml',
            style={'maxWidth': '100%', 'height': 'auto'}
        ),
        style={'width': '100%', 'height': '2500px'}
    )
