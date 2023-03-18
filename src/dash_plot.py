import glob
import os
from itertools import cycle

import matplotlib
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from shapely.geometry import Polygon
import pandas as pd
import numpy as np
from scipy import interpolate
import geopandas as gpd


class DashPlot():
    '''
    This Class Process all plotting activities
    '''

    def __init__(self, project_data=None):
        self.project_data = project_data
        self.marker_symbol_list = cycle([
            'circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'triangle-left',
            'triangle-right', 'pentagon', 'hexagon', 'octagon', 'star', 'circle-cross',
            'circle-x'])
        self.color_list = cycle([
            '#1f77b4',  # muted blue
            '#ff7f0e',  # safety orange
            '#2ca02c',  # cooked asparagus green
            '#d62728',  # brick red
            '#9467bd',  # muted purple
            '#8c564b',  # chestnut brown
            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow-green
            '#17becf'   # blue-teal
        ])
        self.instrument_marker_map = {'CPT': 'circle',
                                      'BH': 'square',
                                      'SCPT': 'star',
                                      'VC': 'hospital'}
        # center of plot for mapbox plot
        self.mapbox_center = {"lat": 22.277470, "lon": 114.062579}

    def find_instruments(self, names: list, path):
        filelist = glob.glob(os.path.join(path, '*'))
        # import pdb; pdb.set_trace()
        file_map = {}
        for name in names:
            file_map[name] = [file for file in filelist if name in file]
        return file_map

    def get_adjacent_GI(self, gdf, center: str, radius=100):
        '''
        This function returns the GI located at a distance away from the target centre.
        Param:
        center: name of the instruments
        radius: the distance around the centre of the target
        '''
        try:
            center_pt = gdf[gdf.ID == center].iloc[0, :].geometry
        except:
            print(f'{center} Key not found in the GeoDataFrame')
        return gdf[gdf.distance(center_pt) < radius]

    def plot_polygon_mapbox(self, gdf, **args):
        '''
        This function plot polygons on the Mapbox Scatter plot.
        Default color of this function is `grey` may need to update this later
        Param: gdf - Geopandas DataFrame object holding the information
        Return: Plotly Figure Object
        '''
        title = ''
        if args.get('title'):
            title = args.get('title')

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
            title=title,
            hovermode='closest',
            showlegend=True,
            margin=dict(l=0, r=0, t=0, b=0),
            mapbox=go.layout.Mapbox(
                accesstoken=self.project_data['mapbox_token'],
                bearing=0,
                center=self.mapbox_center,
                pitch=0,
                zoom=14
            )
        )
        return fig

    def plot_instrument_loc(self, fig=None, plot_portion=True):
        '''
        Plot the locations of instruments, by default the portion geometry will also be ploted.
        '''
        if plot_portion == True:
            assert (self.project_data['portion_geom'].data is not None)
            gdf_portion = self.project_data['portion_geom'].data
            fig = self.plot_polygon_mapbox(gdf_portion)
        else:
            if fig is None:
                fig = go.Figure()
        # 'basic','streets','outdoors', 'light','dark','satellite','satellite-streets'
        gdf_group = self.project_data['instruments_coords'].data.groupby(
            'type')
        for name, group in gdf_group:
            if self.instrument_marker_map.get(name) == None:
                continue
            fig = self.plot_point_mapbox(group, fig, name)
        # style the legend
        fig.update_layout(
            dict(height=600),
            mapbox_style='dark',
            legend=go.layout.Legend(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(
                    family="Arial",
                    size=12,
                    color="#d3d3d3"
                ),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0.4
            ))
        return fig

    def plot_point_mapbox(self, gdf, hoverinfo, instrument_type, fig, **args):
        '''
        Plots scatter plots on mapbox map
        Parameter:
            gdf = GeoDataFrame containing the name and lat lon of the points
            gdf shall have an attribute of `ID` which holds the name of points
            instrument_type: controls the symbols of the markers `CPT`, `SPT`, `VST`
            hoverinfo: the info to show when mouse hovers over the label
            args : will be passed to the scatter plots
        '''
        lat = gdf.geometry.y
        lon = gdf.geometry.x
        hoverinfo = gdf[hoverinfo]
        # color = self.instrument_marker_colors_map[instrument_type]
        color = 'black'
        symbol = self.instrument_marker_map[instrument_type]
        name = instrument_type
        mode = 'markers'
        size = 5
        opacity = 1  # default styling
        if args.get('size') is not None:
            size = args.get('size')
        if args.get('opacity') is not None:
            opacity = args.get('opacity')
        if args.get('color') is not None:
            color = args.get('color')
        if args.get('mode') is not None:
            mode = args.get('mode')
        if args.get('name') is not None:
            name = args.get('name')
        fig.add_trace(go.Scattermapbox(
            lat=lat,
            lon=lon,
            mode=mode,
            text=hoverinfo.values,
            textposition="bottom right",
            hoverinfo='text',
            name=name,
            marker=dict(
                size=size,
                opacity=opacity,
                color=color,
                #                           line=0.2
                symbol=symbol
            )
        )
        )
        fig.update_layout(
            hovermode='closest',
            showlegend=True,
            mapbox=go.layout.Mapbox(
                accesstoken=self.project_data['mapbox_token'],
                bearing=0,
                # style='satellite',
                style='basic',
                center=go.layout.mapbox.Center(
                    lat=22.23,
                    lon=114.3
                ),
                pitch=0,
                zoom=12
            )
        )
        return fig

    def plot_sm_location(self, instrument_type='SM1'):
        '''
        Plots the location of the settlement markers
        '''
        instrument_coords = self.project_data['instruments_coords'].data
        mask = instrument_coords['type'] == instrument_type
        gdf_settlement = instrument_coords[mask]
        fig = self.plot_polygon_mapbox(self.project_data['portion_geom'].data)
        fig = self.plot_point_mapbox(gdf_settlement, fig, instrument_type)
        fig.update_layout(dict(height=600),
                          showlegend=False,
                          mapbox_style='dark')
        return fig

    def get_gdf_coord_from_str(self, instrument_id):
        gdf_instruments_coords = self.project_data['instruments_coords'].data
        mask = gdf_instruments_coords.ID == instrument_id
        return gdf_instruments_coords[mask]

    def read_excel(self, instrument_name, path='./data/sm'):
        files = glob.glob(os.path.join(path, instrument_name)+"*")
        if len(files) > 1:
            files_list = '\n'.join(files)
            warnings.warn(
                f'More than 1 files are found under the {path}, and files are {files_list}, the first one is taken')
        if len(files) == 0:
            warnings.warn(f'No Records {instrument_name} Found!')
            return None
        file = files[0]
        df = pd.read_excel(file).dropna()
        return df

    def get_color_from_val_bound(self, cmap_str: str, min, max, val, n=10):
        bound = np.linspace(min, max, n)
        cmap = matplotlib.cm.get_cmap(cmap_str)
        norm = matplotlib.colors.BoundaryNorm(bound, cmap.N)
        color_R, color_G, color_B, _ = cmap(norm(val))
        return f'rgb({color_R*255:.0f},{color_G*255:.0f}, {color_B*255:.0f})'

    def get_cmap_from_matplotlib(self, cmap_str: str, start_date, end_date, freq='3M', discrete=True):
        '''
        Compute the plotly colormap from matplotlib color map between two dates
        Param:
            cmap_str: a string indicating the color map to be used, e.g., `jet`, `rainbow`
            start_date: datetime object
            end_date: datetime object
            freq: interval of the dates
            discrete: True if discrete color scale is to be used, false otherwise.
        '''
        cmap = matplotlib.cm.get_cmap(cmap_str)
        bound = [x.timestamp() for x in pd.date_range(
            start_date, end_date + np.timedelta64(30, 'D'), freq=freq)]
        norm = matplotlib.colors.BoundaryNorm(bound, cmap.N)
        colormap = []
        tickmark = []
        tickvals = []
        for ix in range(len(bound)-1):
            tickmark.append(pd.to_datetime(
                bound[ix], unit='s').strftime('%Y-%m-%d'))
            tickvals.append(bound[ix])
            if ix % 2 == 1:
                continue
            color_R, color_G, color_B, _ = cmap(norm(bound[ix]))
            colormap.append(
                [norm(bound[ix])/cmap.N, f'rgb({color_R*255:.0f},{color_G*255:.0f}, {color_B*255:.0f})'])
            colormap.append([norm(bound[ix + 1])/cmap.N,
                             f'rgb({color_R*255:.0f},{color_G*255:.0f}, {color_B*255:.0f})'])
    #         import pdb; pdb.set_trace()
        tickmark.append(pd.to_datetime(
            bound[-1], unit='s').strftime('%Y-%m-%d'))
        tickvals.append(bound[-1])
        return colormap, tickmark, tickvals

    def plot_removal_records(self, geojsonfile: str, gdf, z='tsp', date='rmv_date', text='', **args):
        dates = pd.to_datetime(gdf[date])
        start_date = dates.min()
        end_date = dates.max()
        title = ''
        if args.get('title') is not None:
            title = args['title']
        colorscale, tickmark, tickvals = self.get_cmap_from_matplotlib('jet',
                                                                       start_date, end_date, freq='2M')
        gdf_portion = self.project_data['portion_geom'].data
        fig = self.plot_polygon_mapbox(gdf_portion)
        fig.add_trace(go.Choroplethmapbox(geojson=geojsonfile,
                                          locations=gdf.index,
                                          colorscale=colorscale,
                                          z=gdf.tsp,
                                          zmin=gdf.tsp.min(),
                                          zmax=gdf.tsp.max(),
                                          marker_line_width=0.25,
                                          marker_line_color='blue',
                                          marker_opacity=0.8,
                                          text=text,
                                          hoverinfo='text',
                                          colorbar=dict(thickness=20, title="Removal Date",
                                                        titleside='right',
                                                        tickvals=tickvals,
                                                        tickmode='array',
                                                        ticktext=tickmark,
                                                        ticks="inside",
                                                        bgcolor='rgba(0,0,0,0)',
                                                        bordercolor='rgba(0,0,0,0)')))
        fig.update_layout(
            dict(height=800),
            title=title,
            mapbox_style='mapbox://styles/mapbox/satellite-v9',
            mapbox_accesstoken=self.project_data['mapbox_token'],
            mapbox_zoom=14.5,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            mapbox_center=self.mapbox_center,
        )
        return fig

    def calc_rate_of_curve(self, date, data, date_start, date_end):
        min_date = date.min()
        max_date = date.max()
        if date_end < min_date:
            return None
        tsp = date.map(lambda x: x.timestamp())
    #     date_start = date_spot - np.timedelta64(interval_prior, timedelta_type)
    #     date_start = min_date if min_date < date_start else date_start
        date_rng = pd.date_range(start=date_start, end=date_end, freq='1W')
        try:
            s = interpolate.interp1d(
                x=tsp, y=data, fill_value=np.nan, bounds_error=False)
        except:
            return np.nan, np.nan
        date_rng_tsp = date_rng.map(lambda x: x.timestamp())
        interp_data = date_rng_tsp.map(lambda x: s(x))
        df = pd.DataFrame(index=date_rng)
        df['tsp'] = date_rng_tsp
        df['data'] = interp_data
        df.dropna(inplace=True)
    #     do the linear interpolation here
    #     import pdb; pdb.set_trace()
        # time will be in terms of weeks
        x = (df.tsp-df.tsp[0])/(np.timedelta64(1, 'W')/np.timedelta64(1, 's'))
        c1, c0 = np.polyfit(x=x, y=df.data, deg=1)
        return c1, c0

    def offset_data(self, dates, data, offset_date, interval=1, freq='W'):
        '''
        Truncate the data at a given offset date and set the data at `offset_date` as zero
        Param:
            dates : Pandas Series of Dates
            data  : Pandas Series of Data
            interval: interval of interpolation, by default 1 unit
            freq: Frequency for interpolation, by default a week

        '''
        if offset_date < dates.min():
            return pd.DataFrame(dict(data=data), index=dates)
        elif offset_date > dates.max():
            return None
        max_date = dates.max()
        date_rng = pd.date_range(start=offset_date, end=max_date, freq=freq)
        s = interpolate.interp1d(x=dates.map(
            lambda x: x.timestamp()), y=data, fill_value=np.nan, bounds_error=False)
        fitted_value = s(date_rng.map(lambda x: x.timestamp()))
        df = pd.DataFrame(index=date_rng)
        df['data'] = fitted_value - fitted_value[0]
        return df

    def offset_figure(self, fig, offset_date, cut_off_date=False, show_average=True, show_confidence_level=True):
        '''
        Given a figure, and offset the figure at a given offset date
        '''
        fig_offset = go.Figure()
        traces = []
        original_settlement_trace = [
            trace for trace in fig.data if trace.xaxis == 'x2']
        df_data = pd.DataFrame()
        for trace in original_settlement_trace:
            name = trace.name
            x = trace.x
            y = trace.y
            marker_symbol = trace.marker.symbol
            color = trace.marker.color
            df_offset = self.offset_data(
                pd.Series(x), y, offset_date=offset_date)
            if cut_off_date:
                df_offset = df_offset[df_offset.index < cut_off_date]
                end_date = cut_off_date
            else:
                try:
                    end_date = df_offset.index.max()
                except:
                    print(df_offset)
            if df_offset is not None:  # the offset date is greater than the latest monitoring data
                #             import pdb; pdb.set_trace()
                if df_data.shape[0] == 0:
                    df_data = df_offset
                else:
                    df_data = df_data.join(df_offset, rsuffix='_'+name)
                traces.append(go.Scatter(x=df_offset.index, y=df_offset.data,
                                         mode='markers + lines',
                                         name=name,
                                         marker=dict(color=color,
                                                     symbol=marker_symbol)))
                c1, c0 = self.calc_rate_of_curve(
                    df_offset.index, df_offset.data, offset_date, end_date)
            else:  # if offset date greater than the latest monitoring data, not plotting the data
                continue
        # do the statistic analysis here
        df_data['mean'] = df_data.mean(axis=1)
        df_data['std'] = df_data.std(axis=1)
        df_data['low'] = df_data['mean'] - 1.0*df_data['std']
        df_data['height'] = df_data['mean'] + 1.0*df_data['std']
        # add the average
        traces.append(go.Scatter(x=df_data.index, y=df_data['mean'],
                                 mode='markers + lines',
                                 name='mean',
                                 marker=dict(color='blue', symbol='circle-open')))
        for trace in traces:
            fig_offset.add_trace(trace)
        fig_offset.update_yaxes(
            autorange='reversed', title=f"Relative Settlement since {offset_date.strftime('%Y/%m/%d')}")
        return fig_offset, df_data

    def plot_scatter(self, gdf, fig=None, **kwargs):
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        if fig is None:
            fig = go.Figure()
        trace = go.Scatter(x=x, y=y, **kwargs)
        fig.add_trace(trace)
        return fig

    def set_fig_zero_margin(self, fig):
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        return fig

    def set_fig_transprent(self, fig):
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        return fig

    @classmethod
    def gdf_translate_rotate(cls, gdf, **kwargs):
        gdf = gdf.copy()  # we don't want to affect the original data
        xoff = kwargs.get('xoff')
        yoff = kwargs.get('yoff')
        gdf.geometry = gdf.geometry.translate(xoff=xoff, yoff=yoff)
        if kwargs.get('angle') is not None:
            gdf.geometry = gdf.geometry.rotate(
                angle=kwargs['angle'], origin=kwargs['origin'])
        return gdf

    @classmethod
    def voronoi_finite_polygons_2d(cls, vor, radius=2300):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
        Returns
        -------
        gdf_new_regions: GeoDataframe containing the region

        """
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")
        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()*2
        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))
        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]
            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue
            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]
            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue
                # Compute the missing endpoint of an infinite ridge
                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            # finish
            new_regions.append(new_region.tolist())
        # Dazhong Li We will output the GeoDatafram containing only the region, the points and
        new_vertices = np.array(new_vertices)
        geometry = []
        for region in new_regions:
            polygon_vertices = new_vertices[region]
            polygon = Polygon(polygon_vertices)
            geometry.append(polygon)
        gdf_new_regions = gpd.GeoDataFrame(geometry=geometry)
        return gdf_new_regions

    @classmethod
    def calculate_contour_points(cls, gdf, z, nx=100, ny=100, method='linear',
                                 extrapolate=True):
        # convert them to complex number
        nx = complex(0, nx)
        ny = complex(0, ny)
        x_min, x_max = gdf.geometry.x.values.min(), gdf.geometry.x.values.max()
        y_min, y_max = gdf.geometry.y.values.min(), gdf.geometry.y.values.max()
        grid_x, grid_y = np.mgrid[x_min:x_max:nx, y_min:y_max:ny]
        xi = (grid_x, grid_y)
        points = np.array([gdf.geometry.x.values, gdf.geometry.y.values]).T

        z0 = interpolate.griddata(points,
                                  z,
                                  xi, method='nearest',
                                  #               fill_value =nan,
                                  )
        z1 = interpolate.griddata(points,
                                  z,
                                  xi, method=method,
                                  #               fill_value =nan,
                                  )
        if extrapolate:
            z = np.zeros(shape=z0.shape)
            # fill the nans with those from nearest
            z = np.zeros(shape=z0.shape)
            for i in range(z0.shape[0]):
                for j in range(z0.shape[1]):
                    z[i, j] = z0[i, j] if np.isnan(z1[i, j]) else z1[i, j]
        else:
            z = z1
        return grid_x[:, 1], grid_y[1, :], z.T

    @classmethod
    def plot_contour(cls, x, y, z, fig=None, **kwargs):
        if fig is None:  # if we are creating a new figure
            fig = go.Figure()
        trace = go.Contour(
            x=x, y=y, z=z, **kwargs
        )
        fig.add_trace(trace)
        return fig

    @classmethod
    def plot_gdf_polygon(cls, gdf, fig=None, **kwargs):
        if fig is None:
            fig = go.Figure()
        traces = []
        for _, row in gdf.iterrows():
            x = list(row.geometry.exterior.xy[0])
            y = list(row.geometry.exterior.xy[1])
            traces.append(go.Scatter(x=x, y=y, **kwargs))
        for trace in traces:
            fig.add_trace(trace)
        return fig

    @classmethod
    def remove_multipolygon(cls, gdf):
        '''
        removes the mutipolygons
        '''
        for ix, row in gdf.copy().iterrows():
            geom = row['geometry']
            if geom.type == 'MultiPolygon':
                for polygon in geom:
                    add_series = row.copy()
                    add_series.geometry = polygon
                    gdf.append(add_series)
                gdf.drop(ix, axis=0, inplace=True)
        return gdf
