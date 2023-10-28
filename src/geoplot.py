import copy
import json
import logging
import symbol
from itertools import cycle

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as plt
import scipy.stats as st
from plotly.graph_objects import Figure

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
        if size == 'A4' and orientation == 'h':
            fig.update_layout(width=800, height=566, template='simple_white',
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
        for row in range(rows):
            for col in range(cols):
                fig.update_xaxes(showgrid=True, row=row+1, col=col+1)
                fig.update_yaxes(showgrid=True, row=row+1, col=col+1)


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
    def add_marker_plot(cls, x, y, data, hue, window_size=10, fig=None, row=1, col=1, statistic=False):
        '''
        x - label of x value
        y - label of y value
        data - dataframe of plot
        hue - sort
        window_size - size of the window for the moving average (default=10)
        statistic - whether the moving average and confidence interval should be plotted (default=True)
        '''
        if fig is None:
            fig = GEOPlot.get_figure(rows=1, cols=1)
        markers = GEOPlot.marker_list(open_marker=True)
        data = data.sort_values(by=y)
        df_plot = data.groupby(by=hue)
        for key, group in df_plot:
            x_data = group[x]
            y_data = group[y]
            fig.add_trace(go.Scatter(x=x_data, y=y_data, name=key, mode='markers', marker=dict(
                symbol=next(markers))), row=row, col=col)

        # Calculate moving average and 95% confidence interval if statistic=True
        if statistic:
            x_means = []
            x_cis_upper = []
            x_cis_lower = []
            for i in range(len(data)):
                if i < window_size:
                    sub_data = data.iloc[:i+1]
                else:
                    sub_data = data.iloc[i-window_size+1:i+1]
                x_mean = sub_data[x].mean()
                x_ci = st.t.interval(
                    0.95, len(sub_data[x])-1, loc=x_mean, scale=st.sem(sub_data[x]))
                x_means.append(x_mean)
                x_cis_upper.append(x_ci[1])
                x_cis_lower.append(x_ci[0])

            # Add moving average line
            fig.add_trace(go.Scatter(x=x_means, y=data[y], name='Moving Average', mode='lines', line=dict(
                color='black', width=2)), row=row, col=col)

            # Add 95% confidence interval
            fig.add_trace(go.Scatter(x=x_cis_upper, y=data[y], mode='lines', line=dict(
                color='lightblue', width=0), fill='tonextx', showlegend=False), row=row, col=col)
            fig.add_trace(go.Scatter(x=x_cis_lower, y=data[y], mode='lines', line=dict(
                color='lightblue', width=0), fill='tonextx', name='95% CI'), row=row, col=col)

        fig.update_xaxes(title=x, row=row, col=col)
        fig.update_yaxes(title=y, row=row, col=col)
        return fig

    @ classmethod
    def get_figure_from_json(cls, filename):
        '''
        This function returns the figure from a json file
        '''
        with open(filename) as fin:
            x = json.load(fin)
        return go.Figure(x)

    @classmethod
    def plot_distribution(cls, df, col_name):
        """
        Plot distribution of a column in a dataframe and fit a normal distribution to it.

        Args:
        - df: pandas dataframe
        - col_name: name of column to plot

        Returns:
        - fig: plotly figure object
        """
        # Get column data
        col_data = df[col_name]

        # Calculate mean and standard deviation
        col_mean = col_data.mean()
        col_std = col_data.std()

        # Create histogram trace
        hist_trace = go.Histogram(x=col_data, nbinsx=30, name='Histogram',
                                  histnorm='probability density', marker_color='grey')

        # Create normal distribution trace
        x_range = np.linspace(col_data.min()-3*col_std,
                              col_data.max()+3*col_std, num=100)
        y_range = (
            np.exp(-0.5 * ((x_range - col_mean) / col_std) ** 2) /
            (col_std * np.sqrt(2 * np.pi))
        )
        norm_trace = go.Scatter(
            x=x_range, y=y_range, mode='lines', name='Normal Distribution', line_color='blue')

        # Create figure layout
        fig_layout = go.Layout(title=f"Distribution Plot of {col_name}", xaxis_title=col_name, yaxis_title='Probability Density',
                               width=800, height=600, margin=go.layout.Margin(l=50, r=50, b=50, t=50, pad=4),
                               plot_bgcolor='rgb(240,240,240)', paper_bgcolor='rgb(240,240,240)',
                               xaxis=dict(showline=True, linewidth=2,
                                          linecolor='black', mirror=True),
                               yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True))

        # Combine traces and layout into figure
        fig = go.Figure(data=[hist_trace, norm_trace], layout=fig_layout)

        # Add mean and standard deviation annotations to figure
        fig.add_annotation(
            x=col_mean, y=0, text=f"Mean: {col_mean:.2f}", showarrow=False, ax=0, ay=-40)
        fig.add_annotation(x=col_mean+col_std, y=np.interp(col_mean+col_std, x_range,
                           y_range), text=f"std: {col_mean+col_std:.2f}", showarrow=False, ax=0, ay=-40)
        fig.add_annotation(x=col_mean-col_std, y=np.interp(col_mean-col_std, x_range,
                           y_range), text=f"std: {col_mean-col_std:.2f}", showarrow=False, ax=0, ay=-40)

        # Add vertical lines for mean and +/- 1 standard deviation
        fig.add_shape(type='line',
                      x0=col_mean, y0=0, x1=col_mean, y1=max(y_range),
                      line=dict(color='red', dash='dash', width=2))
        fig.add_shape(type='line',
                      x0=col_mean+col_std, y0=0, x1=col_mean+col_std, y1=np.interp(col_mean+col_std, x_range, y_range),
                      line=dict(color='green', dash='dash', width=2))
        fig.add_shape(type='line',
                      x0=col_mean-col_std, y0=0, x1=col_mean-col_std, y1=np.interp(col_mean-col_std, x_range, y_range),
                      line=dict(color='green', dash='dash', width=2))

        # Add vertical lines for mean+/-1std and mean+/-2std
        for i, std in enumerate([2]):
            x_pos = col_mean + (std * col_std)
            y_pos = np.interp(x_pos, x_range, y_range)
            fig.add_annotation(
                x=x_pos, y=y_pos, text=f"{std} Std: {x_pos:.2f}", showarrow=False, ax=0, ay=-40)
            fig.add_shape(type='line',
                          x0=x_pos, y0=0, x1=x_pos, y1=y_pos,
                          line=dict(color=['purple', 'green', 'orange'][i], dash='dot', width=1))
            fig.add_shape(type='line',
                          x0=col_mean - (std * col_std), y0=0, x1=col_mean - (std * col_std), y1=np.interp(col_mean - (std * col_std), x_range, y_range),
                          line=dict(color=['purple', 'green', 'orange'][i], dash='dot', width=1))
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        return fig.update_layout(width=800, height=600)

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
            raise ('Soil Name Not defined')

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
