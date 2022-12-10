import webbrowser
from threading import Timer

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc, html
from app import app
from apps import overall, about

arup_logo = 'logo.png'
navbar = dbc.Navbar(
    children=[
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src=app.get_asset_url(
                        arup_logo), height='40px')),
                    # dbc.Col(dbc.NavbarBrand(
                    #     'Monitoring Data Analyses', className='ml-2')),
                    dbc.Col(dbc.NavLink("Data", href='/')),
                    dbc.Col(dbc.NavLink('Filtering', href='settlement')),
                    dbc.Col(dbc.NavLink('PWP', href='settlement')),
                    # dbc.Col(dbc.NavLink('Progress', href='progress')),
                    dbc.Col(dbc.NavLink('Excel', href='analysis')),
                    dbc.Col(dbc.NavLink('Plaxis', href='plaxis')),
                    dbc.Col(dbc.NavLink('About',
                                        href='report'), className="width400")
                ]
            )
        )
    ],
    sticky="top",
    color='light',
    dark=True,
    expand=True
)
body = dbc.Container([
    html.Div([dcc.Location(id='url', refresh=False)], className='row'),
    html.Div(id='page-content'),
], fluid=True)
app.layout = html.Div([navbar, html.Br(), body])

# --- Callbacks --- #


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return overall.layout
#         # return
#     elif pathname == '/settlement':
#         # importlib.reload(sm)
#         sm.layout = sm.__layout()  # we force recalculation
#         return sm.layout
#     elif pathname == 'surface_movement_marker':
#         return smm.layout
#     elif pathname == 'piezometer':
#         return vwp.layout
#     elif pathname == 'extensometer':
#         return extensometer.layout
#     elif pathname == '/progress':
#         return progress.layout
#     elif pathname == '/analysis':
#         return analysis.layout
#     elif pathname == '/plaxis':
#         return plaxis.layout
    else:
        return about.layout


port = 5000


def open_browser():
    webbrowser.open_new(f'http://localhost:{port}')


if __name__ == '__main__':

    # Timer(1, open_browser).start()
    app.run_server(debug=True, port=port)
