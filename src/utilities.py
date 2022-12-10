import json
import logging
import re
from itertools import cycle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from datetime import datetime
from pathlib import Path
from PyPDF2 import PdfMerger

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
            #logger.debug(f'{col} coversion failed')
            pass
    return df


def read_input_file(filename):
    ''' this function reads a yaml file in and return a dictionary'''

    if filename.endswith('.yaml'):
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
