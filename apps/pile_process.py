import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

pile_dim_card = dbc.Card(
    [
        dbc.CardHeader("Pile Dimension"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Label("Pile Diameter (m)",
                                style={"font-size": "14pt"}), width=6),
                        dbc.Col(dcc.Input(id="input-diameter",
                                type="number", value=0), width=6),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Pile Thickness (mm)",
                                style={"font-size": "14pt"}), width=6),
                        dbc.Col(dcc.Input(id="input-thickness",
                                type="number", value=0), width=6),
                    ],
                    align="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Label("Pile Embedment (m)",
                                style={"font-size": "14pt"}), width=6),
                        dbc.Col(dcc.Input(id="input-embedment",
                                type="number", value=0), width=6),
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
                                "font-size": "14pt"}), width=3),
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
                                style={"font-size": "14pt"}), width=3),
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
                                "font-size": "14pt"}), width=3),
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
        html.H2("Pile Input"),
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
        html.H2("Output"),
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
