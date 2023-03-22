import base64
import io

import matplotlib.pyplot as plt

from UnitCPT import (Input, Output, Path, PipePile, app, dash_table, dbc, dcc,
                     html, pd, px)

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
                                type="number", value=3.5), width=6),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Pile Thickness (mm)",
                                style={"font-size": "12pt"}), width=6),
                        dbc.Col(dcc.Input(id="input-thickness",
                                type="number", value=50), width=6),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Pile Embedment (m)",
                                style={"font-size": "12pt"}), width=6),
                        dbc.Col(dcc.Input(id="input-embedment",
                                type="number", value=60), width=6),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Pile Length (m)",
                                style={"font-size": "12pt"}), width=6),
                        dbc.Col(dcc.Input(id="input-length",
                                type="number", value=70), width=6),
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
sidebar = html.Div(
    [
        html.H4("Input"),
        html.Label("Geological condition"),
        dcc.Dropdown(
            options=[
                {"label": "Location 1", "value": "loc1"},
                {"label": "Location 2", "value": "loc2"},
                {"label": "Location 3", "value": "loc3"},
            ],
            value="loc1",
        ),
        pile_dim_card,
        loading_card,
        pile_soil_card,
        pile_resp_card
    ],
    className="sidebar",
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


@app.callback(
    Output('fig-pile-shape', 'children'),
    [Input('input-diameter', 'value'),
     Input('input-thickness', 'value'),
     Input('input-length', 'value'),
     Input('input-embedment', 'value')
     ]
)
def update_pile_geometry(diameter, thickness, length, penetration):
    pile = PipePile(dia=diameter, thickness=thickness/1000,
                    penetration=penetration, length=length)
    fig, ax = pile.plot()
    resize_figure(fig, 1500, 5000, 600)

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
