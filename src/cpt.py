import copy
import logging
import re
from enum import Enum
from itertools import cycle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import degrees, log10, pi, radians, tan

from .geoplot import GEOPlot
from .ags import AGSParser
from .utilities import to_numeric_all, plot_showgrid
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s -%(pathname)s:%(lineno)d %(levelname)s - %(message)s', '%y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
PA = 101.325


def row_of_key(filename, key):
    o = open(filename, 'r')
    lines = o.readlines()
    for ix, line in enumerate(lines):
        if key in line:
            return ix
    o.close()


class CPTMethod(Enum):
    UWA_05 = 1
    ICP_05 = 2
    FUGRO_05 = 3
    NGI_05 = 4


def extract_data_from(filename, row_number, pattern='ags'):
    # fixed width pattern
    if pattern == 'ags':
        pattern = '^(.{5})(.{9})(.{10})(.{11})(.{11})(.{11})(.{11})'
    if pattern == 'ASCII':
        pattern = r'(.{7})(.{10})(.{10})(.{10})(.{10})'
    else:
        pattern = pattern
    o = open(filename, 'r')
    lines = o.readlines()
    output = []
    for ix in range(row_number+1, len(lines)):
        string = lines[ix]
        try:
            data = re.findall(pattern, string)[0]
            new_line = [x.strip() for x in data]
        except:
            print(f'error on {string}')
            continue
        finally:
            output.append(new_line)
            o.close()
    return output


class CPT:
    name_map = {
        'SCPT_DPTH': 'Depth (m)',
        'SCPT_RES': 'qc (MPa)',
        'SCPT_FRES': 'fs (kPa)',
        'SCPT_PWP2': 'u (kPa)',
        'u0': 'u0 (kPa)',
        'Fr': 'Fr (%)',
        'SPT': 'SPT N60 (blows/30cm)',
        'M': 'Constrained Mod. (MPa)',
        'Dr': 'Dr (%)',
        'phi': 'Friction angle (°)',
        'G0': 'Go (MPa)',
        'qt': 'qt (MPa)',
        'gamma': 'γ (kN/m³)',
        'Ic': 'Ic'
    }
    name_map_reverse = {v: k for k, v in name_map.items()}

    def __init__(self, filename='', key='Data table', pattern='ags', net_area_ratio=0.85):
        if filename != '':
            self.df = self.read_data(filename, key, pattern)
        self.soil_stratum = None
        # Ranges 0.70 ~ 0.85 (page 22 of CPT Guide 2015 Robertson)
        self.net_area_ratio = net_area_ratio
        self.unit_set = False  # Need to set the unit before process

    def read_ASCII(self, filename, key, pattern):
        '''
        Read data from ASCII file
        '''
        key_row = row_of_key(filename, key)
        output = extract_data_from(filename, key_row, pattern)
        with open('cpt_temp.csv', 'w') as fout:
            for ix, line in enumerate(output):
                if ix == 1:  # The first line should be unit and we will ignore
                    continue
                fout.write(','.join(line)+'\n')
        df = pd.read_csv('cpt_temp.csv',
                         engine='python')
        # first is unit
        for c in df.columns:
            df[c] = pd.to_numeric(df[c])
        self.df = df
        return df

    def read_ags(self, filename, unit=['MPa', 'MPa', 'MPa'], ags_format=2):
        '''
        Call the AGSParse module to read the ags data in.
        Delete the first two rows of the data, which are supposed to include the unit
        '''

        with open(filename) as fin:
            ags_str = fin.read()
        ags_parser = AGSParser(ags_str=ags_str, ags_format=ags_format)
        try:
            df = ags_parser.get_df_from_key(ags_parser.keys.SCPT)
            # drop the first two lines that contains unit
            df = df.drop([0, 1], axis=0)
            df = to_numeric_all(df)
            df = df.dropna(how='all', axis=1)
            logger.debug('Import CPT Data')
        except Exception as e:
            logger.error(e)
            raise ('Data Not Imported')
        self.df = df
        self.set_data_unit(unit)
        logger.info(
            f'Imported data include:{df.shape[0]} rows and headers = {list(df.columns)}')
        return ags_parser

    def init_CPT(self):
        '''
        '''
        assert (self.unit_set == True)
        self.df['qt'] = self.df.SCPT_RES + \
            self.df.SCPT_PWP2 * (1-self.net_area_ratio)/1000
        self.df['Rf'] = self.df.SCPT_FRES/self.df.qt/1000*100
        self.df['gamma'] = self.df.apply(lambda row: self.gamma_total(
            row.Rf, row.qt), axis=1)
        self.df['gamma'] = self.df.gamma.fillna(method='ffill')
        self.df['sigma_v'] = ((self.df.SCPT_DPTH.shift(-1) -
                              self.df.SCPT_DPTH).fillna(method='ffill')*self.df.gamma).cumsum()
        self.df['u0'] = self.df.SCPT_DPTH*10
        self.df['sigma_v_e'] = self.df.sigma_v-self.df.u0
        self._cpt_classification()
        logger.debug(
            f'qt calculated using net area ratio of {self.net_area_ratio}')

    def _cpt_classification(self):

        df = self.df
        self.df['qnet'] = self.df['qt'] - self.df['sigma_v']/1000
        # calculate the normalized pore pressure Bq
        self.df['Bq'] = (self.df.SCPT_PWP2 - self.df.u0)/self.df.qnet/1000

        # calculate the 'normalized cone penetration resistance (dimensionless)
        self.df['Qt1'] = (self.df.qt*1000 - self.df.sigma_v)/self.df.sigma_v_e

        # calculate the Fr 'normalized friction ratio in %
        self.df['Fr'] = (self.df.SCPT_FRES /
                         (self.df.qt - self.df.sigma_v/1000))/10
        # Calculate the 'Soil Behavior Type Index' Ic
        self.df['Ic'] = self.df.apply(lambda row: self.calc_Ic(
            row.qt, row.sigma_v, row.sigma_v_e, row.Fr)[0], axis=1)
        self.df['n'] = self.df.apply(lambda row: self.calc_Ic(
            row.qt, row.sigma_v, row.sigma_v_e, row.Fr)[1], axis=1)
        self.df['Qtn'] = (df.qt*1000 - df.sigma_v)/PA * (PA/df.sigma_v_e)**df.n
        self.df['k'] = self.df.apply(
            lambda row: self.permeability(row.Ic), axis=1)
        self.df['M'] = self.df.apply(lambda row: self.constrained_modulus(
            row.Ic, row.Qt1, row.qt, row.sigma_v), axis=1)
        self.calc_relative_density()

    def assign_geological_condition(self, soil_stratum):
        '''
        In a lot of cases, we will need to know the effective stress level at a particular depth
        to understand the soil conditions.  The effective stresses and total stress will be calculated
        Param:
        soil_stratum: a data structure contains the information of the soil layering
        '''
        assert (self.unit_set == True)
        self.soil_stratum = soil_stratum
        if 'SCPT_DPTH' not in self.df.columns:
            raise ('SCPT_DPTH not found, please rename the columns')
        self.df['sigma_v_e'] = self.df.SCPT_DPTH.apply(
            self.soil_stratum.effective_stress)
        self.df['u0'] = self.df.SCPT_DPTH.apply(
            lambda x: 0 if x <= soil_stratum.water_table else (x-soil_stratum.water_table)*10)
        self.df['sigma_v'] = self.df.sigma_v_e + self.df.u0

        # calculate the qt, i.e., the corrected cone resistance
        # calculate the qnet, the difference between the total cone resistance and the toal overburden
        self._cpt_classification()

        self.df = self.soil_stratum.add_soil_names_df(self.df, 'SCPT_DPTH')
        return self.df

    def get_SBTn_plot(self):
        '''
        Plot the soil behaviour Type by Robertson 1990 and  updated in 2010, refer to page 27 of CPT Guide 2015
        '''
        df = self.df
        fig = GEOPlot.get_figure(rows=1, cols=1)
        depth_range = [self.max_depth, 0]
        fig.update_yaxes(range=depth_range)
        tickvals = [0, 1.3, 2.05, 2.6, 2.95, 3.6]
        fig.update_xaxes(tickvals=tickvals)
        color = GEOPlot.get_color()
        for tick in tickvals[:]:
            fig.add_trace(go.Scatter(x=[tick, tick], y=depth_range, showlegend=False,
                                     fill='tonexty'))
        fig.add_trace(go.Scatter(y=df.SCPT_DPTH, x=df.Ic, line=dict(color='black'),
                                 showlegend=False))
        # fig.add_trace(go.Scatter(x=[0.65], y=[np.array(depth_range).mean()],mode='text',text='Sands-clean sand to Silt Sand',
        #  orientation ='v'))
        levels = [0.65, 1.875, 2.525, 2.975, 3.475, 4]
        lables = ['Gravelly sand to dense sand', 'Sands - clean sand to silty sand', 'Sand mixtures - silty sand to sandy silt',
                  'Silt mixture - Clayey silt to silty clay', 'Clays - silty clay to clay', 'Organic Soils - clay']
        for level, text in zip(levels, lables):
            fig.add_annotation(go.layout.Annotation(x=level-0.2, y=np.array(
                depth_range).mean(), text=text, showarrow=False, textangle=-90, font=dict(color='blue')))
        fig.update_xaxes(
            title='Ic (Normalised CPT Soil Behavior Type (SBTn) Chart (Robertson, 1990, updated 2010)', title_font_size=10)
        fig.update_yaxes(title='Depth(m)')
        return fig

    def plot_SBTn_full(self, plotname=''):
        '''
        Plot the SBTn of the data within df
        required column names of the dataframe will be
        ['SCPT_DPTH','SCPT_PWP2','Fr',]
        '''
        df = self.df
        if 'SCPT_DPTH' not in self.df.columns:
            columns = self.df.columns
            updated_columns = [CPT.name_map_reverse[x]
                               if x in CPT.name_map_reverse.keys() else x for x in columns]
            df.columns = updated_columns

        fig = GEOPlot.get_figure(rows=1, cols=4)
        if hasattr(CPT, 'max_depth'):
            max_depth = self.max_depth
        else:
            self.max_depth = self.df.SCPT_DPTH.max()
            max_depth = self.max_depth
        fig.update_yaxes(range=[max_depth, 0], dtick=2)
        fig.update_layout(height=800, width=1200)
        x_labels = ['qt (MPa)', 'Frictional Ratio (%)',
                    'Power Water Pressure (kPa)', 'Soil Behavior Type IC', 'Soil Behavior Type Ic']
        for i in range(4):
            fig.update_yaxes(showgrid=True, title='Depth(m)', col=i+1, row=1)
            fig.update_xaxes(showgrid=True, col=i+1, row=1,
                             side='top', title=x_labels[i])

        fig.update_xaxes(range=[0, 25], dtick=5, row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.qt, y=df.SCPT_DPTH, name='qt(MPa)'), 1, 1)
        fig.add_trace(go.Scatter(x=df.SCPT_PWP2, y=df.SCPT_DPTH,
                                 name='u2(kPa)'), col=3, row=1)
        fig.add_trace(go.Scatter(x=df.u0, y=df.SCPT_DPTH,
                                 name='Hydrostatic(kPa)'), col=3, row=1)
        fig.add_trace(go.Scatter(x=df.Fr, y=df.SCPT_DPTH, name='Fr (%)'), 1, 2)
        fig.update_xaxes(dtick=2, range=[-2, 10], row=1, col=2)
        fig.update_layout(title=plotname)
        fig2 = self.get_SBTn_plot()
        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=4)
        for annotation in fig2.layout.annotations:
            fig.add_annotation(annotation, row=1, col=4)
        return fig

    def plot_su(self, plotname='', nkt=12):
        '''
        '''
        fig = GEOPlot.get_figure(rows=1, cols=4)
        df = self.df
        max_depth = self.max_depth
        fig.update_yaxes(range=[max_depth, 0], dtick=2)
        fig.update_layout(height=800, width=1200)
        x_labels = ['qt (MPa)', 'Vertical Overburden Stress (kPa)',
                    'qnet (MPa)', 'su(kPa)']
        for i in range(4):
            fig.update_yaxes(showgrid=True, title='Depth(m)', col=i+1, row=1)
            fig.update_xaxes(showgrid=True, col=i+1, row=1,
                             side='top', title=x_labels[i])

        fig.update_xaxes(range=[0, 25], dtick=5, row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.qt, y=df.SCPT_DPTH, name='qt(MPa)'), 1, 1)
        fig.add_trace(go.Scatter(x=df.sigma_v, y=df.SCPT_DPTH,
                                 name='simga_v (kPa)'), col=2, row=1)
        fig.add_trace(go.Scatter(x=df.qnet, y=df.SCPT_DPTH,
                                 name='qnet(MPa)'), col=3, row=1)
        fig.add_trace(go.Scatter(x=df.qnet/nkt*1000,
                                 y=df.SCPT_DPTH, name=f'su(Nkt={nkt})'), 1, 4)

        fig.update_xaxes(range=[0, 300], row=1, col=4)
        fig.update_xaxes(range=[0, 25], row=1, col=1)
        fig.update_xaxes(range=[0, 25], row=1, col=3)
        fig.update_layout(title=plotname)
        return fig

    def set_data_unit(self, unit=['MPa', 'MPa', 'MPa']):
        '''
        Some format of data present a different setting of the data unit, we would prefer that
        qc in MPa, fs and PWP in kPa, for the simplicity, we will need to the key to be consistent
        with the ags format, i.e., SCPT_RES, SCPT_FRES and SCPT_PWP2
        '''
        if self.unit_set == False:
            if unit[0] == 'kPa':
                self.df.SCPT_RES = self.df.SCPT_RES/1000
            if unit[1] == 'MPa':
                self.df.SCPT_FRES = self.df.SCPT_FRES*1000
            if unit[2] == 'MPa':
                self.df.SCPT_PWP2 = self.df.SCPT_PWP2*1000
            self.unit_set = True
        self.max_depth = np.ceil((self.df['SCPT_DPTH'].max())/5+1)*5
        logger.debug(f'Data unit is set to {unit}')

    def update_data_column_name(self, columns=[]):
        '''
        The name of columns should be
        SCPT_DPTH - Depth of Cone
        SCPT_FRES - Shaft Resistance
        SCPT_RES - Cone Resistance
        SCPT_PWP2 - Porewater pressure measured at the sensor measured at the shoulder of the cone
        '''
        self.df.columns = columns
        self.max_depth = np.ceil((self.df['SCPT_DPTH'].max())/5+1)*5

    def cal_qt(self, qc, u, a=0.8):
        '''
        Returns the corrected cone resistance for pore water pressure
        '''
        return qc + (1-a) * u

    @classmethod
    def peak_friction_angle(cls, qc, sigma_v):
        '''
        Returns the peak friction angle based on Roberson & Campanella (1983)
        This only apply for sand
        '''
        return degrees(1/2.68 * (log10(qc/sigma_v*1000)+0.29))

    @classmethod
    def add_SBT_axes(cls, fig, depth_range, row=1, col=1):
        '''
        Add the label and ticks for soil classification for plotting Ic
        fig - A plotly figure
        depth_range: range of the depth to be plotted, e.g., [50,0]
        row, col = row and col for multi plots
        '''
        fig.update_yaxes(range=depth_range, row=row, col=col)
        tickvals = [0, 1.3, 2.05, 2.6, 2.95, 3.6, 4]
        fig.update_xaxes(tickvals=tickvals, row=row, col=col)
        color = cycle([
            'rgba(63, 123, 26, 0.3)',  # Not used
            'rgba(59, 225, 93, 0.5)',  # Gravel
            'rgba(0, 178, 47, 0.3)',   # Clean Sand
            'rgba(63, 123, 26, 0.3)',  # Sand
            'rgba(114, 58, 164, 0.4)',  # Silty
            'rgba(61, 58, 79, 0.5)',  # Silt Clay
            'rgba(0,68,48, 0.5)',  # Clay
        ])

        for tick in tickvals[:]:
            fig.add_trace(go.Scatter(x=[tick, tick], y=depth_range, showlegend=False,
                                     fill='tonexty', mode='lines', fillcolor=next(color), line=dict(color='black', width=0.2)), row=row, col=col)
        # fig.add_trace(go.Scatter(x=[0.65], y=[np.array(depth_range).mean()],mode='text',text='Sands-clean sand to Silt Sand',
        #  orientation ='v'))
        levels = [0.65, 1.875, 2.525, 2.975, 3.475, 4]
        labels = ['Gravelly sand to dense sand', 'Sands - clean sand to silty sand', 'Sand mixtures - silty sand to sandy silt',
                  'Silt mixture - Clayey silt to silty clay', 'Clays - silty clay to clay', 'Organic Soils - clay']
        for level, text in zip(levels, labels):
            fig.add_annotation(go.layout.Annotation(x=level-0.2, y=np.array(
                depth_range).mean(), text=text, textangle=-90, font=dict(color='rgba(0,0,0,0.8)')), row=row, col=col)

    def plot_ags_cpt(self, fig=None, water_level=0, name='', **kwargs):
        '''
        This function plots the CPT data extracted in the ags format
        '''
        df = self.df
        if fig is None:
            fig = GEOPlot.get_figure(rows=1, cols=3, **kwargs)
        if 'SCPG_TESN' in df.columns:
            df_group = df.groupby('SCPG_TESN')
        else:
            df_group = zip(name, df)
        for name, group in df_group:
            fig.add_trace(go.Scatter(y=group.SCPT_DPTH, x=group.SCPT_RES,
                                     showlegend=False, name=name, line=dict(color='black')), 1, 1)
            fig.add_trace(go.Scatter(y=group.SCPT_DPTH, x=group.SCPT_FRES,
                                     showlegend=False, name=name, line=dict(color='black')), 1, 2)
            fig.add_trace(go.Scatter(y=group.SCPT_DPTH, x=group.SCPT_PWP2,
                                     showlegend=False, name=name, line=dict(color='black')), 1, 3)

            # Let's determine the range of data
            max_depth = self.max_depth
            for i in range(1, 4):
                fig.update_yaxes(range=[max_depth, 0], dtick=2,
                                 title='Depth below seabed(m)', row=1, col=i)
            fig.update_xaxes(title='qc(MPa)', side='top', row=1, col=1)
            fig.update_xaxes(title='Shaft Friction(kPa)',
                             side='top', row=1, col=2, showgrid=True)
            fig.update_xaxes(title='Pore Water Pressure u2(kPa)',
                             side='top', row=1, col=3, showgrid=True)
            fig.update_layout(width=1200, height=800, title=name)
        if water_level is not None:
            hydrostatic = np.where(
                df.SCPT_DPTH >= water_level, 10*(df.SCPT_DPTH-water_level), 0)
            fig.add_trace(go.Scatter(
                x=hydrostatic, y=df.SCPT_DPTH, name='Hydrostatic'), 1, 3)
        fig = plot_showgrid(fig, 3)
        return fig

    def get_Dr_plot(self, plotname='', method='Jamiolkowski'):
        self.calc_relative_density(method=method)
        fig = GEOPlot.get_figure()
        tickvals = [0, 15, 30, 56, 80, 100]
        for tick in tickvals[:]:
            fig.add_trace(go.Scatter(x=[tick, tick], y=[0, self.max_depth], showlegend=False,
                                     fill='tonexty'))
        df = self.df
        depth_range = [self.max_depth, 0]
        fig.update_yaxes(range=depth_range)
        fig.update_xaxes(range=[0, 100])
        fig.add_trace(go.Scatter(x=df.Dr, y=df.SCPT_DPTH, line=dict(
            color='black'), showlegend=False, name='Dr'))
        fig.update_xaxes(tickvals=tickvals,
                         title='Relative Density (Dr %)', side='top')
        fig.update_yaxes(title='Depth(m)')
        levels = [8, 25, 40, 70, 90]
        texts = ['Very Loose', 'Loose', 'Medium Dense', 'Dense', 'Very Dense']
        for level, text in zip(levels, texts):
            fig.add_annotation(go.layout.Annotation(x=level, y=np.array(depth_range).mean(), xref='x', yref='y', text=text,
                                                    textangle=-90, font=dict(color='blue')))
        fig.update_layout(title=plotname)
        return fig

    def calc_relative_density(self, method='Jamiolkowski'):
        '''
        Calculate the relative density Dr. based on the CPT data
        Jamiolkowski, M., Lo Presti, D. C. F., & Manassero, M. (2003).
        Evaluation of relative density and shear strength of sands from CPT and DMT.
        In Soil behavior and soft ground construction (pp. 201-238).
        qt: in MPa
        sigma_v: in kPa

        0~15 very loose
        15~35 Loose
        35~65 Medium
        65~85 Dense
        85~100 Very Dense
        '''
        qt = self.df.qt
        sigma_v = self.df.sigma_v
        qt = qt * 1000  # convert to kPa
        pa = 101.325  # atmospheric pressure
        qt1 = (qt/pa) / ((sigma_v/pa)**0.5)

        if method == 'Jamiolkowski':
            bx = 0.675
            Dr = 100 * (0.268 * np.log(qt1) - bx)

        if method == 'Claussen':
            Dr = 0.4*np.log(qt1/22)
        self.df['Dr'] = Dr
        return self.df

    def _shaft_friction(self, z,  qt, sigma_v_e, delta, pile, Ic, method=CPTMethod.UWA_05, b_compression=True, consider_clay=False):
        '''
        This returns unit shaft friction of a pile using the CPT data
        qt - cone resistance, we should fit qt in
        sigma_v_e - effective vertical stress in kPa
        pile - the pile class hold the basic information of pile
        delta - in degrees, the delta_cv, the constant volume friction between the pile and soil
        method: "UWA" - method 2 in API
                "ICP" - Method 1 in API
        '''
        R_star = np.sqrt((pile.dia_out/2)**2 - (pile.dia_inner/2)**2)
        qt = qt*1000  # qc in MPa
        h = pile.embedment-z
        if Ic >= 2.8:  # we are dealing with clay
            if not consider_clay:
                pass
            else:  # Lehane (2013)
                return 0.055 * qt*np.max([h/R_star, 1])**(-0.2)

        delta = radians(delta)
        if method == CPTMethod.UWA_05:
            a = 0
            b = 0.3
            c = 0.5
            d = 1
            e = 0
            v = 2.0
            u = 0.03 if b_compression else 0.022
        elif method == CPTMethod.ICP_05:
            a = 0.1
            b = 0.2
            c = 0.4
            d = 1.0
            e = 0
            v = 4*(pile.disp_ratio)**0.5
            u = 0.023 if b_compression else 0.016
        elif method == CPTMethod.FUGRO_05:  # method 3
            a = 0.05 if b_compression else 0.15
            b = 0.45 if b_compression else 0.42
            c = 0.90 if b_compression else .85
            d = 0.0
            e = 1.0 if b_compression else 0
            u = 0.043 if b_compression else 0.025
            v = 2*(pile.disp_ratio)**0.5

        elif method == CPTMethod.NGI_05:  # method 4
            pass
        else:
            print('Not Implemented')
        Ar = pile.disp_ratio
        D = pile.dia_out
        L = pile.embedment
        depth_ratio = (L-z)/D
        c1 = u*qt*(sigma_v_e/100)**a
        c2 = max(depth_ratio, v)**(-c)
        c3 = min(depth_ratio/v, 1)**e
        fs = c1 * Ar**b * c2 * tan(delta)**d * c3
        return fs

    def _base_Qb(self, qc, qc_average, pile, Dr, method=CPTMethod.UWA_05):
        pa = 101
        Ar = pile.disp_ratio
        qc = 1000*qc
        qc_average = 1000*qc_average
        D = pile.dia_out
        Di = pile.dia_inner
        Dcpt = 0.036  # for a standard cone with base area of 10cm2
        if method == CPTMethod.UWA_05:
            q = qc_average * (0.15 + 0.45 * Ar)
            return q * pile.gross_area
        elif method == CPTMethod.ICP_05:
            q = max((0.5-0.25*np.log10(D/Dcpt)), 0.15)*qc_average
            if Di > 2*(Dr - 0.3) or Di/Dcpt > 0.083*qc/pa:
                return qc * pile.annulus_area
            else:
                return pile.gross_area * q
        elif method == CPTMethod.FUGRO_05:
            q = 8.5*100*(qc_average/100)**0.5*Ar**0.25
            return q*pile.gross_area
        else:
            print('Not Implemented!')

    def shaft_friction(self, pile, delta_cv=28.5, method=CPTMethod.UWA_05, compression=True, consider_clay=False):
        '''
        This function returns the unit shaft friction along the pile
        Qf = diameter * tau(z)
        '''

        self.df['shaft_F'] = pile.perimeter_outer * self.df.apply(lambda row: self._shaft_friction(
            row.SCPT_DPTH, row.qt, row.sigma_v, delta_cv, pile, row.Ic, method, compression, consider_clay=consider_clay), axis=1)
        self.df['Qs'] = (self.df.SCPT_DPTH.shift(-1) -
                         self.df.SCPT_DPTH)*self.df.shaft_F.cumsum()
        return self.df

    def base_force(self, pile, method=CPTMethod.UWA_05):
        df = self.df
        increment = df.SCPT_DPTH[1] - df.SCPT_DPTH[0]
        D = pile.dia_out
        # we are going to do the averaging using 3D window, i.e., 1.5D above and below the toe level
        window = int(3*D/increment)
        df['qt_avg'] = df.qt.rolling(window=window, center=True).mean()
        df['Qb'] = df.apply(lambda row: self._base_Qb(
            row['qt'], row['qt_avg'], pile, row['Dr'], method=method), axis=1)
        return df

    def calc_Ic(self, qt, sigma_v, sigma_v_e, Fr):
        n0 = 1
        pa = 100  # 100kPa
        qt = qt*1000  # qt is in MPa
        n_iter = 1
        while n_iter < 200:
            Qtn = ((qt-sigma_v)/pa)*(pa/sigma_v_e)**n0
            Ic = ((3.47-np.log10(Qtn))**2+(np.log10(Fr)+1.22)**2)**0.5
            if n_iter == 1:
                Ic1 = Ic
                Qt1 = Qtn
            n1 = 0.381*(Ic) + 0.05*(sigma_v_e/pa) - 0.15
            if abs(n1-n0) < 1.0e-3:
                if n1 >= 1:
                    return Ic1, 1, Qt1
                else:
                    return Ic, n0, Qtn
            else:
                n0 = n1
                n_iter = n_iter + 1
        return np.nan, np.nan

    def permeability(self, Ic):
        if Ic < 3.27 and Ic > 1:
            return 10**(0.952-3.04*Ic)
        if Ic <= 4.0 and Ic > 3.27:
            return 10**(-4.52-1.37*Ic)
        else:
            return np.nan

    def constrained_modulus(self, Ic, Qt1, qt, sigma_v):
        if Ic > 2.0:
            aM = min(Qt1, 14)
        else:
            aM = 0.0188*(10**(0.55*Ic+1.68))
        return aM*(qt-sigma_v/1000)

    def gamma_total(self, Rf, qt):
        '''
        Rf : Ratio of the shaft friction to the corrected cone resistance, i.e., fs/qt
        '''
        return 9.8*((0.27*np.log10(Rf)) + 0.36*(np.log10(qt*1000/PA)) + 1.236)

    @classmethod
    def plot_qc(cls, df, qc='qc (MPa)', depth='Depth (m)', Ic='Ic', fig=None, row=1, col=1):
        if fig is None:
            fig = GEOPlot.get_figure()
        fig.add_trace(go.Scatter(x=np.where(df[Ic] < 0.65, df[qc], 0), y=df[depth], mode='none',
                                 fill='tozerox', fillcolor='rgba(59, 225, 93, 1.0)',
                                 name='Gravelly sand to dense sand'), row=row, col=col)
        fig.add_trace(go.Scatter(x=np.where((df[Ic] >= 0.65) & (df.Ic < 1.875), df[qc], 0), y=df[depth], mode='none',
                                 fill='tozerox', fillcolor='rgba(0, 178, 47, 0.3)',
                                 name='Sands - clean sand to silty sand'), row=row, col=col)
        fig.add_trace(go.Scatter(x=np.where((df[Ic] > 1.875) & (df.Ic < 2.525), df[qc], 0), y=df[depth], mode='none',
                                 fill='tozerox', fillcolor='rgba(63, 123, 26, 0.3)',
                                 name='Sand mixtures - silty sand to sandy silt'), row=row, col=col)
        fig.add_trace(go.Scatter(x=np.where((df[Ic] > 2.525) & (df.Ic < 2.975), df[qc], 0), y=df[depth], mode='none',
                                 fill='tozerox', fillcolor='rgba(114, 58, 164, 0.4)',
                                 name='Silt mixture - Clayey silt to silty clay'), row=row, col=col)
        fig.add_trace(go.Scatter(x=np.where((df[Ic] > 2.975) & (df.Ic < 3.475), df[qc], 0), y=df[depth], mode='none',
                                 fill='tozerox', fillcolor='rgba(61, 58, 79, 0.5)',
                                 name='Clays - silty clay to clay'), row=row, col=col)
        fig.add_trace(go.Scatter(x=np.where(df[Ic] > 3.475, df[qc], 0), y=df[depth], mode='none',
                                 fill='tozerox', fillcolor='rgba(0,68,48, 0.5)',
                                 name='Organic Soils - clay'), row=row, col=col)
        # fig.add_trace(go.Scatter(x=np.where(df.Ic>3,df['qc (MPa)'],0),y=df['Depth (m)'],mode='none',fill='tozerox'))
        fig.add_trace(go.Scatter(x=df[qc], y=df[depth], mode='lines', line=dict(
            color='black', width=1), name='qc'), row=row, col=col)
        fig.update_yaxes(range=[30, 0], title='Depth (m)')
        fig.update_xaxes(range=[0, 50], title='qc (MPa)', side='top')
        fig.update_layout(legend=dict(orientation='h'), width=800, height=800)
        return fig

    @classmethod
    def plot_qc_matplotlib(cls, df, qc='qc (MPa)', depth='Depth (m)', Ic='Ic', ax=None):
         if ax is None:
             fig, ax = plt.subplots(figsize=(8, 8))
         # Fill the area under the curve with different colors based on the value of Ic
         ax.fill_betweenx(df[depth], np.where(df[Ic] < 0.65, df[qc], 0), 0,
                          color=(59/255, 225/255, 93/255, 0.9), label='Gravelly sand to dense sand', step='pre', interpolate=True, alpha=0.2)
         ax.fill_betweenx(df[depth], np.where((df[Ic] >= 0.65) & (df.Ic < 1.875), df[qc], 0), 0,
                          color=(0/255, 178/255, 47/255, 0.3), label='Sands - clean sand to silty sand', step='pre', interpolate=True, alpha=0.2)
         ax.fill_betweenx(df[depth], np.where((df[Ic] > 1.875) & (df.Ic < 2.525), df[qc], 0), 0,
                          color=(63/255, 123/255, 26/255, 0.3), label='Sand mixtures - silty sand to sandy silt', step='pre', interpolate=True, alpha=0.2)
         ax.fill_betweenx(df[depth], np.where((df[Ic] > 2.525) & (df.Ic < 2.975), df[qc], 0), 0,
                          color=(114/255, 58/255, 164/255, 0.4), label='Silt mixture - Clayey silt to silty clay', step='pre', interpolate=True, alpha=0.2)
         ax.fill_betweenx(df[depth], np.where((df[Ic] > 2.975) & (df.Ic < 3.475), df[qc], 0), 0,
                          color=(61/255, 58/255, 79/255, 0.5), label='Clays - silty clay to clay', step='pre', interpolate=True, alpha=0.2)
         ax.fill_betweenx(df[depth], np.where(df[Ic] > 3.475, df[qc], 0), 0,
                          color=(0/255, 68/255, 48/255, 0.5), label='Organic Soils - clay', step='pre', interpolate=True, alpha=0.2)
         # Plot the qc curve
         ax.plot(df[qc], df[depth], '-k', linewidth=1, label='qc')
         ax.invert_yaxis()
         ax.set_ylim([30, 0])
         ax.set_xlim([0, 60])
         ax.set_xlabel('qc (MPa)')
         ax.set_ylabel('Depth (m)')
         ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=2)

         return ax.get_figure(), ax

    def calc_CSR(self, PGA):
        '''
        calculate the cyclic stress ratio based on Seed and Idriss (1971) using a simplified method
        CSR = tau_max/sigma_v_e = 0.65* PGA/g *(sigma_v/sigma_v_e)* rd
        where `rd` is stress reduction factor, which can be calculated using hte tri-linear function 
        rd = 1.0-0.00765z (for z < 9.15)
        rd = 1.174 - 0.0267z ( for z between 9.15 to 23m)
        rd = 0.744 - 0.008 z ( for z between 23 to 30m )
        rd = 0.5 if z > 30
        '''
        def rd(z):
            if z < 9.15:
                return 1.0-0.00765*z
            elif z >= 9.15 and z < 23:
                return 1.174 - 0.0267*z
            elif z >= 23 and z < 30:
                return 0.744 - 0.008*z
            else:  # blow 30m
                return 0.5
        self.df['CSR'] = 0.65*PGA * \
            (self.df.sigma_v/self.df.sigma_v_e)*self.df.SCPT_DPTH.apply(rd)

    def _Kc(self, Ic):
        '''
        Todo: need to consider the impact of the Fr see page 113 of the Robertson
        '''
        if Ic <= 1.64:
            return 1.0
        else:
            return 5.581*Ic**3 - 0.403*Ic**4 - 21.63*Ic**2 + 33.75*Ic - 17.88

    def calc_Qtncs(self):
        '''
        Kc - correlation factor that relates the Qtn_cs (for clean sand to the calculated Qtn)
        '''
        df = self.df
        df['Kc'] = df.Ic.apply(self._Kc)
        df['Qtncs'] = df.Kc * df.Qtn

    def calc_CRR75(self):
        '''
        calculated the cyclic resistance adjusted to M=7.5 earthquake
        '''
        def _CRR75(Qtncs):
            '''
            calculate the cyclic resistance adjusted to M=7.5 earthquake 
            following Robertson and Wride (1998)
            '''
            if Qtncs < 50:
                return 0.833 * (Qtncs/1000) + 0.05
            else:
                return 93 * (Qtncs/1000)**3 + 0.08
        self.df['CRR75'] = self.df.Qtncs.apply(_CRR75)

    def calc_liquefaction_FOS(self, PGA=0.2, M=7.5):
        '''
        calculate the FOS against liquefaction
        '''
        self.calc_Qtncs()
        self.calc_CSR(PGA)
        self.calc_CRR75()
        MSF = 174/(M**2.56)  # Magnitude Scaling Factor
        self.df['FoS_Liq'] = self.df.CRR75/self.df.CSR * MSF
        logger.debug(f'Liquefaction FOS calculated for PGG={PGA}g and M={M}')

    def __str__(self):
        if self.df is None:
            return 'No Data is associated with object'
        else:
            print_str = f'''
            alpha = {self.net_area_ratio}
            no. Data = {self.df.shape[0]}
            current_columns = {self.df.columns}
            '''
            return print_str
