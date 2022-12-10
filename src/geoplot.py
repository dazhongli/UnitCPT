import copy
import json
import logging
from itertools import cycle
import symbol

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as plt
from plotly.graph_objects import Figure
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class GEOPlot():
    def __init__():
        pass

    @classmethod
    def get_figure(cls, size='A4', orientation='v', transparent=False, cols=1, rows=1, logo=True, footer=None):
        fig = plt.make_subplots(cols=cols, rows=rows)
        if size == 'A4' and orientation == 'v':
            fig.update_layout(width=566, height=800, template='simple_white',
                              margin=dict(l=100, r=20, t=60, b=150),
                              xaxis=dict(showgrid=True),
                              yaxis=dict(showgrid=True),
                              )
        if transparent:
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)')
        if logo:
            fig.layout.images = [dict(
                source="https://gitlab.com/dazhong.li/utility/-/blob/master/image/arup_logo.png",
                xref="paper", yref="paper",
                x=0.1, y=1.0,
                sizex=0.2, sizey=0.3,
                xanchor="center", yanchor="bottom"
            )]
        if footer is not None:
            fig.layout.annotations = [
                go.layout.Annotation(
                    showarrow=False,
                    text=footer,
                    xanchor='right',
                    xref='paper', yref='paper',
                    x=1,
                    yanchor='top', y=0.01,
                    font=dict(size=4)
                )]
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))

        return fig

    @classmethod
    def get_data(cls, fig, row, col):
        if row == 1 and col == 1:
            index = ''
        index = row*col
        for ix, data in enumerate(fig.data):
            if data.xaxis == f'x{index}':
                return ix, data

    @classmethod
    def insert_figure(cls, fig_to: Figure, t_row: int, t_col: int,
                      fig_from: Figure, f_row: int, f_col: int, delete_original=True) -> Figure:
        '''
        This function insert a subplot from a figure to another one.
        Param:
        fig_to : to hosting figure to accept the subplot
        t_row:   row in the hosting figure
        t_col:   col in the hosting figure
        '''
        fig_to = copy.copy(fig_to)
        fig_from = copy.copy(fig_from)
        if delete_original:
            cls.remove_data_from_axis(fig_to, t_row, t_col, inplace=True)
        from_fig_ID = f_row*f_col
        row = t_row
        col = t_col
        # Let's format the two figures for the easier handling
        for data in fig_from.data:
            if data.xaxis is None:
                data.xaxis = 'x'
                data.yaxis = 'y'
                from_fig_ID = ''

        for data in fig_from.data:
            if data.xaxis == f'x{from_fig_ID}' and data.yaxis == f'y{from_fig_ID}':
                fig_to.add_trace(data, row=row, col=col)
        for annot in fig_from.layout.annotations:
            if annot.xref == f'x{from_fig_ID}' and annot.yref == f'y{from_fig_ID}':
                fig_to.add_annotation(annot, row=row, col=col)

        xaxis = fig_from.layout[f'xaxis{from_fig_ID}']
        xaxis.domain = fig_to.layout[f'xaxis{row*col}'].domain
        yaxis = fig_from.layout[f'yaxis{from_fig_ID}']
        # yaxis.domain = fig_to.layout[f'yaxis{}'].domain
        fig_to.update_xaxes(xaxis, row=row, col=col)
        # fig_to.update_yaxes( #     fig_from.layout[f'yaxis{from_fig_ID}'], row=row, col=col)
        return fig_to

    @classmethod
    def remove_data_from_axis(cls, fig, row, col, inplace=False):
        '''
        This function removes all data from one subplot defined by (row, col)
        '''
        if not inplace:
            fig = copy.copy(fig)
        for data in fig.data:
            if (data.xaxis is None):
                data.xaxis = 'x'
                data.yaxis = 'y'
        list_to_remove = []
        for ix, data in enumerate(fig.data):
            if row == 1 and col == 1:
                if data.xaxis == 'x':
                    list_to_remove.append(ix)
            if data.xaxis == f'x{row}':
                list_to_remove.append(ix)
        else:
            for ix, data in enumerate(fig.data):
                if data.xaxis == f'x{row*col}':
                    list_to_remove.append(ix)
        for ix in list_to_remove:
            fig.data[ix].x = []
        # let's remove annotation if any
        return fig

    @classmethod
    def add_marker_plot(cls, x, y, data, hue, fig=None, row=1, col=1):
        '''
        x - label of x value
        y - label of y value
        data - dataframe of plot
        hue - sort
        '''
        if fig is None:
            fig = cls.get_figure(rows=1, cols=1)
        markers = cls.marker_list(open_marker=True)
        df_plot = data.groupby(by=hue)
        for key, group in df_plot:
            x_data = group[x]
            y_data = group[y]
            fig.add_trace(go.Scatter(x=x_data, y=y_data, name=key, mode='makers', maker=dict(
                symbol=next(markers))), row=1, col=1)
        fig.update_xaxes(title=x)
        fig.update_yaxes(title=y)
        return fig

    @ classmethod
    def get_figure_from_json(cls, filename):
        '''
        This function returns the figure from a json file
        '''
        with open(filename) as fin:
            x = json.load(fin)
        return go.Figure(x)

    @ classmethod
    def get_color(cls, color_type='Plotly'):
        '''
        valid color type include ['Plotly','D3','G10', 'T10','Alphabet']
        '''
        return cycle(getattr(px.colors.qualitative, color_type))

    @ classmethod
    def get_soil_color(cls, soil_name='CLAY'):
        '''
        Accepted Soil Names:
        'CLAY', "GRAVEL",'SAND','SILT'
        '''

        if soil_name == 'CLAY':
            return 'rgba(0,68,48, 0.5)'  # Clay
        if soil_name == 'GRAVEL':
            return 'rgba(59, 225, 93, 0.5)'  # Gravel
        if soil_name == 'SAND':
            return 'rgba(0, 178, 47, 0.3)'   # Clean Sand
        if soil_name == 'SILT':
            return 'rgba(61, 58, 79, 0.5)'
        else:
            raise('Soil Name Not defined')

    @classmethod
    def marker_list(cls, open_marker=False):
        if open_marker:
            marker_list = cycle(['circle-open', 'square-open', 'diamond-open',
                                 'hexagram-open', 'diamond-cross-open', 'diamond-x-open', 'hash-open',
                                 'circle-x-open', 'hash-open', 'star-square-open',
                                 'diamond-tall-open', 'diamond-wide-open', 'hourglass-open',
                                 'bowtie-open', 'hexagon-open', 'hexagon2-open'])
        else:
            marker_list = cycle(['circle', 'square', 'diamond', 'cross',
                                 'hexagram', 'diamond-cross', 'diamond-x',
                                 'asterisk', 'hash',
                                 'circle-x', 'hash', 'star-square',
                                 'diamond-tall', 'diamond-wide', 'hourglass',
                                 'bowtie', 'hexagon', 'hexagon2', 'y-up'])
        return marker_list

    @ classmethod
    def plot_voronoi_graph(cls, gdf_vor_regions, fig=None, row=1, col=1):
        if fig is None:
            fig = go.Figure()
        traces = []
        for _, row in gdf_vor_regions.iterrows():
            polygon = row['geometry']
            x = np.array(polygon.exterior.coords.xy)[0, :]
            y = np.array(polygon.exterior.coords.xy)[1, :]
            traces.append(go.Scatter(x=x, y=y,
                                     mode='lines',
                                     showlegend=False,
                                     hoverinfo='none',
                                     line=dict(color='black',
                                               width=0.5,
                                               dash='dot')))
        for trace in traces:
            fig.add_trace(trace, row=1, col=2)
        return fig


# color_list = cycle(['black', 'blue', 'green', 'red', 'yellow'])
# color_list = cycle(['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])

def plot(df, x, y, **argv):
    fig = GEOPlot.get_figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], **argv))
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return fig
