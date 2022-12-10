import dash
import dash_bootstrap_components as dbc
app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.JOURNAL])  # Dark theme
server = app.server
app.config.suppress_callback_exceptions = True
app.title = 'Probabilistic Seismic Hazard Assessment'
