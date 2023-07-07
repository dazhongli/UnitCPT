import json
import logging
import re
from datetime import datetime
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from PyPDF2 import PdfMerger

from geoplot import GEOPlot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s -%(pathname)s:%(lineno)d %(levelname)s - %(message)s', '%y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __getitem__(self, key):
        return self.__getattribute__(key)


def to_numeric_all(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            # logger.debug(f'{col} coversion failed')
            pass
    return df


def read_input_file(filename):
    ''' this function reads a yaml file in and return a dictionary'''

    if filename.endswith('.yaml') or filename.endswith('.yml'):
        return _read_yaml_file(filename)
    if filename.endswith('.json'):
        return _read_json_file(filename)


def interp_row(df, val, by_col, method='ffill', sort='ascending'):
    '''
    This function inserts a row with value of 'val', at the column 'by_col', other values in the dataframe will be initiated as
    np.nan and then interpolated or filled
    '''
    # check if the value has already in the dataframe
    if np.any((df[by_col]-val).abs() <= 1e-6):
        return df  # don't need to do anything
    df_insert = df.copy()
    # df_insert = df.set_index(by_col)
    df_insert.loc[len(df)] = np.nan
    df_insert.loc[-1, by_col] = val
    if sort == 'ascending':
        df_insert.sort_values(by=by_col, ascending=True, inplace=True)
    else:
        df_insert.sort_values(by=by_col, ascending=False, inplace=True)
    df_insert.fillna(method=method, inplace=True)
    return df_insert.reset_index(drop=True).dropna()
    # return df_insert


def _read_yaml_file(filename):
    with open(filename) as fin:
        file = fin.read()
        param = yaml.load(file, Loader=yaml.FullLoader)
        return param
    # return Struct(**_convert_data(param))


def _read_json_file(filename):
    with open(filename, 'r') as fin:
        data = json.load(fin)
    return _convert_data(data)


def _convert_data(data):
    for key in data:
        if isinstance(data[key], dict):
            try:
                data[key] = pd.DataFrame(data[key])
            except:
                data[key] = pd.DataFrame(data[key], index=[0])
        elif isinstance(data[key], list):
            data[key] = np.array(data[key])
    return data


def get_date_time():
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return date_time


def merge_pdf(pdf_path, name):
    date_time = get_date_time()
    pdf_path = Path(pdf_path)
    pdf_files = list(pdf_path.glob('*.pdf'))
    merger = PdfMerger()
    for pdf in pdf_files:
        merger.append(pdf)
    merger.write(f'./{date_time}-{name}.pdf')
    merger.close()


def plot_showgrid(fig, naxis=1):
    '''
    show the grid of all axis
    '''
    for i in range(1, naxis+1):
        if i == 1:
            fig.layout.xaxis.showgrid = True
            fig.layout.yaxis.showgrid = True
        else:
            fig.layout[f'xaxis{i}'].showgrid = True
            fig.layout[f'yaxis{i}'].showgrid = True
    return fig


def convert_to_geojson(cls, gdf, filename, suffix='_wgs84', init_epsg='epsg:2326'):
    '''saves two copies of geojson file, one with the original coordinates system
    the otherone is the WGS84 format'''
    assert ('.json' in filename)
    gdf.crs = {'init': init_epsg}
    gdf.to_file(filename, driver='GeoJSON')
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    gdf_wgs84.to_file(filename.split(
        '..json')[0] + '_wgs84' + '.json', driver='GeoJSON')

# ---------------------------------------[plot] ----------------------------------------------


def plot_point_mapbox(gdf, x, y, text, fig=None, **argv):
    '''
    Plots the scatter plot on a mapbox
    @Param:
    gdf: GeoDataFrame
    '''
    lat = gdf.geometry.y
    lon = gdf.geometry.x
    hoverinfo = gdf[text]
    if fig is None:
        fig = GEOPlot.get_figure()
    fig.add_trace(go.Scattermapbox(
                  lat=lat,
                  lon=lon,
                  mode='markers',
                  text=hoverinfo.values,
                  textposition="bottom right",
                  hoverinfo='text',
                  name=instrument_type,
                  marker=dict(
                      size=5,
                      opacity=1,
                      **argv,
                  )
                  )
                  )
    fig.update_layout(
        hovermode='closest',
        showlegend=True,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            style='satellite',
            center=go.layout.mapbox.Center(
                lat=22.277470,
                lon=114.062579
            ),
            pitch=0,
        ))


def plot_polygon_mapbox(gdf):
    fig = go.Figure()
    for ix, row in gdf.iterrows():
        x = list(row.geometry.exterior.xy[0])
        y = list(row.geometry.exterior.xy[1])
        fig.add_trace(go.Scattermapbox(
            lat=y,
            lon=x,
            mode='lines',
            hoverinfo='none',
            showlegend=False,
            marker=go.scattermapbox.Marker(color='grey',
                                           size=0.01
                                           ),
            text=['Hong Kong'],
        ))
    fig.update_layout(
        hovermode='closest',
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=22.277470,
                lon=114.062579
            ),
            pitch=0,
            zoom=14
        )
    )
    return fig
