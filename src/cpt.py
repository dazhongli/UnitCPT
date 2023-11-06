import copy
import logging
import re
from enum import Enum
from itertools import cycle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import degrees, log10, pi, radians, tan
from scipy.interpolate import interp1d

from .geoplot import GEOPlot
from .ags import AGSParser
from .utilities import to_numeric_all, plot_showgrid
import matplotlib.pyplot as plt

from .pysand import calc_phi_e, calc_C1, calc_C2, calc_C3,calc_k, calc_pr, calc_A, interpolate_cpt_data, determine_soil_type, calc_y, calc_p, export_p_y_monotonic, export_p_y_cyclic, plot_p_y_curve
from .pyclay import calc_su, calc_su1, calc_alpha, calc_N_pd, calc_d, calc_N_p0, calc_N_P, calc_N_P_anisotropy, calc_pu, calc_OCR, identify_clay_layers, calc_y_mo, calc_p_mo, calc_h_f, calc_N_eq, calc_p_y_mod, calc_p_y_mod, plot_p_y_cyclic

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer

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
    UNIFIED = 5

class Clay_type(Enum):
    Gulf_of_Mexico = 1
    North_Sea_soft_clay = 2
    North_Sea_stiff_clay = 3


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
    
    def load_data(self, df):
        self.df = df

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
        '''
        Backfill the missing values (SCPT_RES, SCPT_FRES, SCPT_PWP2, Rf) in the original cpt file at initial penetration depth. 
        '''
        self.df['SCPT_RES'] = self.df['SCPT_RES'].fillna(method='bfill')
        self.df['SCPT_FRES'] = self.df['SCPT_FRES'].fillna(method='bfill')
        self.df['SCPT_PWP2'] = self.df['SCPT_PWP2'].fillna(method='bfill')
        self.df['qt'] = self.df.SCPT_RES + \
            self.df.SCPT_PWP2 * (1-self.net_area_ratio)/1000
        self.df['Rf'] = self.df.SCPT_FRES/self.df.qt/1000*100
        self.df['Rf'] = self.df.Rf.fillna(method='bfill')
        self.df['gamma'] = self.df.apply(lambda row: self.gamma_total(
            row.Rf, row.qt), axis=1)
        #self.df['sigma_v'] = ((self.df.SCPT_DPTH.shift(-1) -
                              #self.df.SCPT_DPTH).fillna(method='ffill')*self.df.gamma).cumsum()
        self.df['sigma_v'] = self.df.SCPT_DPTH*self.df.gamma
        self.df['u0'] = self.df.SCPT_DPTH*9.8
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
        self.df['Ic'] = self.df.Ic.fillna(method='bfill')
        self.df['n'] = self.df.apply(lambda row: self.calc_Ic(
            row.qt, row.sigma_v, row.sigma_v_e, row.Fr)[1], axis=1)
        self.df['n'] = self.df.n.fillna(method='bfill')
        self.df['Qtn'] = (df.qt*1000 - df.sigma_v)/PA * (PA/df.sigma_v_e)**df.n
        self.df['Qtn'] = self.df.Qtn.fillna(method='bfill')
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
            lambda x: 0 if x <= soil_stratum.water_table else (x-soil_stratum.water_table)*9.8)
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
                df.SCPT_DPTH >= water_level, 9.8*(df.SCPT_DPTH-water_level), 0)
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


    def _shaft_friction_unified_clay(self, pile, z,  qt, Fr, sigma_v, sigma_v_e, Ic):
        '''
        This returns unit shaft friction of a pile at clay layer using unified CPT method
        '''
        n = 0.381 * Ic + 0.05 * (sigma_v_e / PA) - 0.15
        if n>1:
            n=1
        Qtn = (qt*1000 - sigma_v) / PA * (PA / sigma_v_e) ** n
        Iz1 = Qtn - 12 * np.exp(-1.4 * Fr)
        if Iz1 > 0:
            Fst = 1.0
        else:
            Fst = 0.5
        
        D_star = np.sqrt((pile.dia_out)**2 - (pile.dia_inner)**2)
        qt = qt*1000  # qc in MPa
        h = pile.penetration-z

        return 0.07 * Fst * qt * np.max([h/D_star, 1])**(-0.25)
    

    def _shaft_friction_unified_sand(self, pile, z, qc, sigma_v_e, compression=True):
        '''
        This returns unit shaft friction of a pile at sand layer using unified CPT method
        '''
        delta = radians(29)
        Ar = 1 - (np.tanh(0.3 * (pile.dia_inner / 0.0356)**0.5)) * (pile.dia_inner / pile.dia_out)**2
        D = pile.dia_out
        h= pile.penetration-z
        depth_ratio = h/D
        qc = qc*1000  # qc in MPa

        sigma_rc = qc / 44 * Ar ** (0.3) * max(depth_ratio, 1)** (-0.4)
        delta_sigma = qc / 10 * (qc / sigma_v_e) ** (-0.33) * 0.0356 / D

        if compression == True:
            fl = 1.0
        else:
            fl = 0.75

        return fl * (sigma_rc + delta_sigma) * tan(delta)


    def base_Qb_unified_clay(self, pile, qp_average):
        D_star = np.sqrt((pile.dia_out)**2 - (pile.dia_inner)**2)
        Ar = 1 - (np.tanh(0.3 * (pile.dia_inner / 0.0356)**0.5)) * (pile.dia_inner / pile.dia_out)**2
        qp_average = 1000*qp_average
        return (0.2 + 0.6 * (D_star / pile.dia_out) ** 2) * qp_average * pile.gross_area
    

    def base_Qb_unified_sand(self, pile, qp_average):
        Ar = 1 - (np.tanh(0.3 * (pile.dia_inner / 0.0356)**0.5)) * (pile.dia_inner / pile.dia_out)**2
        qp_average = 1000*qp_average
        return (0.12 + 0.38 * Ar) * qp_average * pile.gross_area
    

    def calc_pile_capacity(self, pile, compression = True, plot_fig = False):
        '''
        This function returns the bearing capacity along the pile
        '''
        df = self.df
        increment = df.SCPT_DPTH[1] - df.SCPT_DPTH[0]
        #increment = df.SCPT_DPTH.diff().mean()
        df ['qt_avg_sand'] = df.qt.rolling(window = int(3.0*pile.dia_out/increment), center=True, min_periods=1).mean()
        reversed_df = pd.DataFrame()
        reversed_df ['qt'] = df.qt.iloc[::-1]
        rolling_mean = reversed_df.qt.rolling(window=int(20*pile.t/increment), min_periods=1).mean()
        df ['qt_avg_clay'] = rolling_mean.iloc[::-1]
        df ['Qb'] = df.apply(lambda row:self.base_Qb_unified_sand(pile = pile, qp_average = row['qt_avg_sand']) if row.Ic_predict <2.6 else self.base_Qb_unified_clay(pile = pile, qp_average = row['qt_avg_clay']), axis=1)
        df ['Qb_sand'] = df.apply(lambda row:self.base_Qb_unified_sand(pile = pile, qp_average = row['qt_avg_sand']), axis=1)
        df ['Qb_clay'] = df.apply(lambda row:self.base_Qb_unified_clay(pile = pile, qp_average = row['qt_avg_clay']), axis=1)
        df ['shaft_F'] = pile.perimeter_outer * df.apply(lambda row: self._shaft_friction_unified_sand(
            pile, row.SCPT_DPTH, row.SCPT_RES, row.sigma_v_e, compression) if row.Ic_predict <2.6 else self._shaft_friction_unified_clay(pile, row.SCPT_DPTH, row.qt, row.Fr, row.sigma_v, row.sigma_v_e, row.Ic), axis=1)
        df ['delta_Qs'] = (df.SCPT_DPTH.shift(-1) - df.SCPT_DPTH).fillna(method = 'ffill')*df.shaft_F
        df ['Qs'] = df.delta_Qs.cumsum()
        df ['Qc'] = df ['Qs'] + df ['Qb']

        if plot_fig:
            fig = GEOPlot.get_figure(rows=1, cols=5)
            #fig.add_trace(go.Scatter(x=df.qt,y=df.SCPT_DPTH),row=1,col=1)
            fig.update_yaxes(range=[df.SCPT_DPTH.max(),0], dtick=2)
            fig.update_xaxes(side='top')
            fig.update_layout(height=800, width=1600)

            x_labels = ['qt (MPa)', 'Shaft friction (kN)',
                    'End bearing capacity (kN)', 'Pile axial capaity (kN)', 'Soil layer classification']
            for i in range(5):
                fig.update_yaxes(showgrid=True, title='Depth (m)', col=i+1, row=1)
                fig.update_xaxes(showgrid=True, col=i+1, row=1,
                             side='top', title=x_labels[i])
            #fig.update_xaxes(range=[0, 25], dtick=5, row=1, col=1)
            fig.update_xaxes(dtick=5, row=1, col=1)
            fig.add_trace(go.Scatter(x=df.qt, y=df.SCPT_DPTH, name='qt(MPa)'), col=1, row=1)
            fig.add_trace(go.Scatter(x=df.Qs, y=df.SCPT_DPTH, name='Qs (kN)'), col=2, row=1)
            fig.add_trace(go.Scatter(x=df.Qb, y=df.SCPT_DPTH, name='Qb (kN)'), col=3, row=1)
            fig.add_trace(go.Scatter(x=df.Qc, y=df.SCPT_DPTH, name='Qc (kN)'), col=4, row=1)
            fig.add_trace(go.Scatter(x=df.Ic_predict, y=df.SCPT_DPTH, name='Ic_predict', mode='lines', line=dict(width=3, color='brown')), col=5, row=1)
            fig.add_trace(go.Scatter(x=df.Ic, y=df.SCPT_DPTH, name='Ic_measured', mode='lines', line=dict(width=0.5, color='black')), col=5, row=1)
            fig.add_shape(type='line', x0=2.525, y0=0, x1=2.525, y1=df.SCPT_DPTH.max(), line=dict(width=3, color='dimgray'), col=5, row=1)

            fig.show()

        return df
    
    def calc_p_y_curve(self, pile, compression = True, monotonic = True, isotropy = True, interval = 1.0, y_range = 600, plot_fig = False, Clay_type = Clay_type.Gulf_of_Mexico):
        '''
        This function returns the p-y-cutves along the pile
        '''
        #specified soil parameters
        nkt = 12
        N1 = 12
        N2 = 3.22
        I_p = 35

        #interpolate cpt data and identify soil type
        resampled_cpt = self.interpolate_data(interval)
        #df_resampled = interpolate_cpt_data(self.df, interval)
        df_resampled = resampled_cpt.df
        df_resampled ['soil_type'] = df_resampled.apply(lambda row:determine_soil_type(ic = row['Ic_predict']), axis = 1)

        #calculate p-y curve parameters for sand
        df_resampled ['phi_e'] = df_resampled.apply(lambda row:calc_phi_e(qt = row['qt'], sigma_v = row['sigma_v'], sigma_v_e = row['sigma_v_e']) if row['soil_type'] == 'sand' else None, axis = 1)
        df_resampled ['C1'] = df_resampled.apply(lambda row:calc_C1(phi_e = row['phi_e']) if row['soil_type'] == 'sand' else None, axis = 1)
        df_resampled ['C2'] = df_resampled.apply(lambda row:calc_C2(phi_e = row['phi_e']) if row['soil_type'] == 'sand' else None, axis = 1)
        df_resampled ['C3'] = df_resampled.apply(lambda row:calc_C3(phi_e = row['phi_e']) if row['soil_type'] == 'sand' else None, axis = 1)
        df_resampled ['k'] = df_resampled.apply(lambda row:calc_k(phi_e = row['phi_e']) if row['soil_type'] == 'sand' else None, axis = 1)
        df_resampled ['pr'] = df_resampled.apply(lambda row:calc_pr(D = pile.dia_out, gamma = row['gamma'], z = row['SCPT_DPTH'], C1 = row['C1'], C2 = row['C2'], C3 = row['C3']) if row['soil_type'] == 'sand' else None, axis = 1)
        df_resampled ['A'] = df_resampled.apply(lambda row:calc_A(D = pile.dia_out, z = row['SCPT_DPTH'], monotonic = monotonic) if row['soil_type'] == 'sand' else None, axis = 1)

        #calculate p-y curve parameters for clay
        df_resampled ['su'] = df_resampled.apply(lambda row:calc_su(qt = row['qt'], sigma_v = row['sigma_v'], nkt = nkt) if row['soil_type'] == 'clay' else None, axis = 1)
        identify_clay_layers(df_resampled)
        df_resampled ['su1'] = df_resampled.apply(lambda row:calc_su1(su = row['su'], su0 = row['su0'], z= row['SCPT_DPTH']) if row['soil_type'] == 'clay' else None, axis = 1)
        df_resampled ['alpha'] = df_resampled.apply(lambda row:calc_alpha(su = row['su'], sigma_v_e = row['sigma_v_e']) if row['soil_type'] == 'clay' else None, axis = 1)
        df_resampled ['N_pd'] = df_resampled.apply(lambda row:calc_N_pd(alpha = row['alpha']) if row['soil_type'] == 'clay' else None, axis = 1)
        df_resampled ['d'] = df_resampled.apply(lambda row:calc_d(su0 = row['su0'], su1 = row['su1'], D = pile.dia_out) if row['soil_type'] == 'clay' else None, axis = 1)
        df_resampled ['N_p0'] = df_resampled.apply(lambda row:calc_N_p0(N_1 = N1, N_2 = N2, alpha = row['alpha'], d = row['d'], D = pile.dia_out, N_pd = row['N_pd'], z = row['SCPT_DPTH']) if row['soil_type'] == 'clay' else None, axis = 1)
        df_resampled ['N_P'] = df_resampled.apply(lambda row:calc_N_P(N_pd = row['N_pd'], N_p0 = row['N_p0'], gamma = row['gamma'], z = row['SCPT_DPTH'], su = row['su']) if row['soil_type'] == 'clay' else None, axis = 1)
        N_P0 = df_resampled.loc[df_resampled['soil_type'] == 'clay', 'N_P'].iloc[0]
        df_resampled ['N_P_ani'] = df_resampled.apply(lambda row:calc_N_P_anisotropy(N_P = row['N_P'], N_P0 = N_P0, N_pd = row['N_pd'], N_p0 = row['N_p0'], gamma = row['gamma'], z = row['SCPT_DPTH'], su = row['su']) if row['soil_type'] == 'clay' and isotropy != True else None, axis = 1)
        df_resampled ['pu'] = df_resampled.apply(lambda row:calc_pu(su = row['su'], D = pile.dia_out, N_P = row['N_P']) if row['soil_type'] == 'clay' and isotropy == True else calc_pu(su = row['su'], D = pile.dia_out, N_P = row['N_P_ani']) if row['soil_type'] == 'clay' and isotropy != True else None, axis = 1)
        df_resampled ['OCR'] = df_resampled.apply(lambda row:calc_OCR(Qt1 = row['Qt1']) if row['soil_type'] == 'clay' else None, axis = 1)

        #generate monotonic p-y curves
        for i in range(12):
            df_resampled [f'y{i}'] = df_resampled.apply(lambda row:calc_y_mo(I_p = I_p, OCR = row['OCR'], D= pile.dia_out, id_p = i) if row['soil_type'] == 'clay' else calc_y(y_range = y_range, i = i), axis = 1)
            df_resampled [f'p{i}'] = df_resampled.apply(lambda row:calc_p_mo(pu = row['pu'], id_p = i) if row['soil_type'] == 'clay' else calc_p(y = row[f'y{i}'], A = row['A'], pr = row['pr'], z = row['SCPT_DPTH'], k = row['k']), axis = 1)

        #generate cyclic p-y curves
        if monotonic != True:
            for i in range(12):
                df_resampled [f'h_f{i}'] = df_resampled.apply(lambda row:calc_h_f(p_mo = row[f'p{i}'], p_u = row['pu'], z= row['SCPT_DPTH'], D = pile.dia_out) if row['soil_type'] == 'clay' else None, axis = 1)
                df_resampled [f'N_eq{i}'] = df_resampled.apply(lambda row:calc_N_eq(h_f = row[f'h_f{i}'], clay_type = Clay_type) if row['soil_type'] == 'clay' else None, axis = 1)
                df_resampled [f'p_cy{i}'] = df_resampled.apply(lambda row:calc_p_y_mod(N_eq = row[f'N_eq{i}'], clay_type = Clay_type)[0] * row[f'p{i}'] if row['soil_type'] == 'clay' else None, axis = 1)
                df_resampled [f'y_cy{i}'] = df_resampled.apply(lambda row:calc_p_y_mod(N_eq = row[f'N_eq{i}'], clay_type = Clay_type)[1] * row[f'y{i}'] if row['soil_type'] == 'clay' else None, axis = 1) 

        #export p-y curve to excel
        if monotonic == True:
            export_p_y_monotonic(df_resampled, "cpt_data_p_y_curve.xlsx")
        else:
            export_p_y_cyclic(df_resampled, "cpt_data_p_y_curve_cyclic.xlsx")

        df_resampled.to_excel("cpt_data_resampled.xlsx", index=False)

        #plot p-y curve figures if applicable
        if plot_fig:         
            plot_p_y_curve(df_resampled, 8)
            if monotonic != True:
                plot_p_y_cyclic(df_resampled, 8)

        return df_resampled


    def interpolate_data(self, interval):
        new_depths = pd.Series(np.arange(int(self.df['SCPT_DPTH'].min())+interval, int(self.df['SCPT_DPTH'].max()), interval))
        col_names = self.df.iloc[:, 2:].columns.tolist()
        df_resampled = pd.DataFrame()
        df_resampled['SCPT_DPTH'] = new_depths
        for col_name in col_names:
            func_name = interp1d(x = self.df['SCPT_DPTH'], y = self.df[col_name], kind = 'linear')
            df_resampled[col_name] = func_name(new_depths)
        resampled_cpt = CPT()
        resampled_cpt.load_data(df_resampled)
        return resampled_cpt
    

    def identify_soil_layers(self, num_layers = 15, plot_fig = False):
        X = self.df ['SCPT_DPTH']
        y = self.df ['Ic']

        X = X.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        line_X = np.linspace(X.min(), X.max(), 1000, endpoint=False).reshape(-1, 1)

        enc = KBinsDiscretizer(n_bins = num_layers, encode = "onehot")
        #enc = KBinsDiscretizer(n_bins = num_layers, encode = "ordinal")
        X_binned = enc.fit_transform(X)
        Line_binned = enc.fit_transform(line_X)

        LinearR = LinearRegression().fit(X_binned, y)
        TreeR = DecisionTreeRegressor(random_state = 0).fit(X_binned, y)

        layer_id = []
        depth_binned = []
        Ic_predict = []
        depth_layered = []
        Ic_layered = []
        id_layered = []
        df_layered = pd.DataFrame()

        for i, row in enumerate(Line_binned):
            index = np.argmax(row)
            layer_id = np.append(layer_id, index)
            depth_binned = np.append(depth_binned, line_X[i])
            Ic_predict = np.append(Ic_predict, TreeR.predict(Line_binned)[i])

        for id in range(num_layers):
            id_layered = np.append(id_layered, id)
            indices = np.where(layer_id == id)[0]
            last_row = np.max(indices)
            depth_layered = np.append(depth_layered, depth_binned[last_row])
            Ic_layered = np.append(Ic_layered, Ic_predict[last_row])

        df_layered['id_layered'] = id_layered
        df_layered['depth_layered'] = depth_layered
        df_layered['Ic_layered'] = Ic_layered

        self.df['layer_id'] = self.df['SCPT_DPTH'].apply(
            lambda x: df_layered.loc[df_layered['depth_layered'] >= x, 'id_layered'].iloc[0] 
            if any(df_layered['depth_layered'] >= x) 
            else df_layered['id_layered'].iloc[-1])
        self.df['Ic_predict'] = self.df['SCPT_DPTH'].apply(
            lambda x: df_layered.loc[df_layered['depth_layered'] >= x, 'Ic_layered'].iloc[0] 
            if any(df_layered['depth_layered'] >= x) 
            else df_layered['Ic_layered'].iloc[-1])

        if plot_fig:
            fig = go.Figure()

            ic_value1 = 0.65
            ic_value2 = 1.875
            ic_value3 = 2.525
            ic_value4 = 2.975
            ic_value5 = 3.475

            fig.add_trace(go.Scatter(x=y.flatten(), y=X[:, 0], mode='lines', line=dict(width=0.5, color='black'), name='CPT data'))
            #fig.add_trace(go.Scatter(x=LinearR.predict(Line_binned), y=line_X.flatten(), mode='lines', line=dict(width=5, color='green'), name='linear regression'))
            fig.add_trace(go.Scatter(x=TreeR.predict(Line_binned), y=line_X.flatten(), mode='lines', line=dict(width=5, color='crimson'), name='Fitting with {} soil layers'.format(num_layers)))

            fig.add_shape(type='line', x0=ic_value1, y0=X.min(), x1=ic_value1, y1=X.max(), line=dict(width=1.5, color='sienna'))
            fig.add_shape(type='line', x0=ic_value2, y0=X.min(), x1=ic_value2, y1=X.max(), line=dict(width=1.5, color='palevioletred'))
            fig.add_shape(type='line', x0=ic_value3, y0=X.min(), x1=ic_value3, y1=X.max(), line=dict(width=3, color='dimgray'))
            fig.add_shape(type='line', x0=ic_value4, y0=X.min(), x1=ic_value4, y1=X.max(), line=dict(width=1.5, color='goldenrod'))
            fig.add_shape(type='line', x0=ic_value5, y0=X.min(), x1=ic_value5, y1=X.max(), line=dict(width=1.5, color='cornflowerblue'))

            for y in enc.bin_edges_[0]:
                fig.add_shape(type='line', x0=0, y0=y, x1=y.max(), y1=y, line=dict(width=1, color='silver'))

            fig.add_shape(type='rect', x0=0, y0=X.min(), x1=ic_value1, y1=X.max(), fillcolor='sienna', opacity=0.2, layer='below', line=dict(width=0))
            fig.add_shape(type='rect', x0=ic_value1, y0=X.min(), x1=ic_value2, y1=X.max(), fillcolor='palevioletred', opacity=0.2, layer='below', line=dict(width=0))
            fig.add_shape(type='rect', x0=ic_value2, y0=X.min(), x1=ic_value3, y1=X.max(), fillcolor='grey', opacity=0.2, layer='below', line=dict(width=0))
            fig.add_shape(type='rect', x0=ic_value3, y0=X.min(), x1=ic_value4, y1=X.max(), fillcolor='goldenrod', opacity=0.2, layer='below', line=dict(width=0))
            fig.add_shape(type='rect', x0=ic_value4, y0=X.min(), x1=ic_value5, y1=X.max(), fillcolor='cornflowerblue', opacity=0.2, layer='below', line=dict(width=0))

            fig.add_annotation(x=(0+ic_value1)/2, y=(X.min()+X.max())/2, text='Gravelly sand to dense sand', showarrow=False, font=dict(size=16, color='black'), textangle=-90, align='center', valign='middle')
            fig.add_annotation(x=(ic_value1+ic_value2)/2, y=(X.min()+X.max())/2, text='Sands - clean sand to silty sand', showarrow=False, font=dict(size=16, color='black'), textangle=-90, align='center', valign='middle')
            fig.add_annotation(x=(ic_value2+ic_value3)/2, y=(X.min()+X.max())/2, text='Sand mixtures - silty sand to sandy silt', showarrow=False, font=dict(size=16, color='black'), textangle=-90, align='center', valign='middle')
            fig.add_annotation(x=(ic_value3+ic_value4)/2, y=(X.min()+X.max())/2, text='Silt mixture - Clayey silt to silty clay', showarrow=False, font=dict(size=16, color='black'), textangle=-90, align='center', valign='middle')
            fig.add_annotation(x=(ic_value4+ic_value5)/2, y=(X.min()+X.max())/2, text='Clays - silty clay to clay', showarrow=False, font=dict(size=16, color='black'), textangle=-90, align='center', valign='middle')
            fig.add_annotation(x=(ic_value5+4)/2, y=(X.min()+X.max())/2, text='Organic Soils - clay', showarrow=False, font=dict(size=16, color='black'), textangle=-90, align='center', valign='middle')

            fig.update_xaxes(range=[0, 4], title='Ic', linecolor='black', tickcolor='black', ticks="outside", side='top')
            fig.update_yaxes(range=[X.max(), 0], title='Depth (m)', linecolor='black', tickcolor='black', ticks="outside")
            fig.update_layout(height=750, width=500, margin=dict(l=50, r=50, b=50, t=50, pad=4), plot_bgcolor='white', legend=dict(yanchor='top', xanchor='center', x = 0.5, y = -0.02))
            fig.show()


    def identify_soil_layers_plotly(self, num_layers = 15):
        X = self.df ['SCPT_DPTH']
        y = self.df ['Ic']

        X = X.values.reshape(-1, 1)
        y = y.values.reshape(-1, 1)
        line_X = np.linspace(X.min(), X.max(), 1000, endpoint=False).reshape(-1, 1)

        enc = KBinsDiscretizer(n_bins = num_layers, encode = "onehot")
        #enc = KBinsDiscretizer(n_bins = num_layers, encode = "ordinal")
        X_binned = enc.fit_transform(X)
        Line_binned = enc.fit_transform(line_X)

        LinearR = LinearRegression().fit(X_binned, y)
        TreeR = DecisionTreeRegressor(random_state = 0).fit(X_binned, y)

        fig, ax1 = plt.subplots(figsize = (4, 7.5))

        ic_value1 = 0.65
        ic_value2 = 1.875
        ic_value3 = 2.525
        ic_value4 = 2.975
        ic_value5 = 3.475

        #ax2.plot(LinearR_.predict(Line_binned), line_X, linewidth = 1, color = 'green', linestyle = '-', label = 'linear regression')
        ax1.plot(TreeR.predict(Line_binned), line_X, linewidth = 4, color = 'crimson', linestyle = '-', label = 'decision tree')
        ax1.vlines(ic_value3, X.min(), X.max(), linewidth = 2.5, color='dimgray', label='Ic = {}'.format(ic_value3))
        ax1.vlines(ic_value1, X.min(), X.max(), color='sienna')
        ax1.vlines(ic_value2, X.min(), X.max(), color='palevioletred')
        ax1.vlines(ic_value4, X.min(), X.max(), color='goldenrod')
        ax1.vlines(ic_value5, X.min(), X.max(), color='cornflowerblue')
        ax1.hlines(enc.bin_edges_[0], 0, y.max(), linewidth = 1, alpha = 0.2)
        ax1.plot(y, X[:, 0], linewidth=0.5, color='black')
        ax1.legend(loc = "best")
        ax1.set_xlabel("Ic")
        ax1.set_ylabel("Depth")

        ax1.axvspan(0, ic_value1, X.min(), X.max(), alpha=0.2, color='sienna')
        ax1.axvspan(ic_value1, ic_value2, X.min(), X.max(), alpha=0.2, color='palevioletred')
        ax1.axvspan(ic_value2, ic_value3, X.min(), X.max(), alpha=0.2, color='grey')
        ax1.axvspan(ic_value3, ic_value4, X.min(), X.max(), alpha=0.2, color='goldenrod')
        ax1.axvspan(ic_value4, ic_value5, X.min(), X.max(), alpha=0.2, color='cornflowerblue')

        ax1.text((0 + ic_value1) / 2, (X.min() + X.max()) / 2, "Gravelly sand to dense sand", rotation=90, ha='center', va='center', fontsize = 12, fontname = 'Arial')
        ax1.text((ic_value1 + ic_value2) / 2, (X.min() + X.max()) / 2, "Sands - clean sand to silty sand", rotation=90, ha='center', va='center', fontsize = 12, fontname = 'Arial')
        ax1.text((ic_value2 + ic_value3) / 2, (X.min() + X.max()) / 2, "Sand mixtures - silty sand to sandy silt", rotation=90, ha='center', va='center', fontsize = 12, fontname = 'Arial')
        ax1.text((ic_value3 + ic_value4) / 2, (X.min() + X.max()) / 2, "Silt mixture - Clayey silt to silty clay", rotation=90, ha='center', va='center', fontsize = 12, fontname = 'Arial')
        ax1.text((ic_value4 + ic_value5) / 2, (X.min() + X.max()) / 2, "Clays - silty clay to clay", rotation=90, ha='center', va='center', fontsize = 12, fontname = 'Arial')
        ax1.text((ic_value5 + 4) / 2, (X.min() + X.max()) / 2, "Organic Soils - clay", rotation=90, ha='center', va='center', fontsize = 12, fontname = 'Arial')

        ax1.set_ylim(0, X.max())
        ax1.set_xlim(0, 4.0)

        plt.show()