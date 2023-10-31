import win32com.client
#import geoplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import psutil
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
class ALP:
    def __init__(self, AlP_STR="alpLib_20_0.AlpAuto"):
        self._close_alp()
        self.proj = win32com.client.Dispatch("alpLib_20_0.AlpAuto")
        
    def _close_alp(self):
        '''
        close the instance of the running alp, this will cause error when openning or start a new project
        '''
        for proc in psutil.process_iter():
            try:
            # Get process details as a named tuple
                process_info = proc.as_dict(attrs=['pid', 'name'])
                # print(process_info)
                if process_info['name'] == 'alp.exe':
                    psutil.Process(process_info['pid']).terminate()
                    logger.debug(f"Terminated - {process_info['name']} Pid-{process_info['pid']}")
                # Check if the process name matches the target applicatio
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    def open(self, filename):
        '''
        Open and existing file
        @filename - absolute path of the filename is required
        '''
        if os.path.exists(filename):   
            try:
                self.proj.Open(filename)
                logger.debug(f'Sucessfully started {filename}')
            except Exception as e:
                print(e)
            self.n_nodes = self.proj.GetNumNodes()
            self.df = pd.DataFrame()
            y = np.array([float(self.proj.GetNodeLevel(i)) for i in range(1,self.n_nodes)])
            self.df['y'] = y
        else:
              logger.error(f"File {filename} does not exist.")
    def get_nodal_displacement_BM(self):
        '''
        Return a dataframe containing the displacement of the beam
        '''
        self.df_disp = pd.DataFrame(columns=['node_ID','Level','Disp','BM'])
        node_disp =[]
        for i in range(1,self.n_nodes):
            self.df_disp.loc[i,'node_ID'] = i
            i_disp = self.proj.GetNodeDisp(i)
            BM = self.proj.GetNodeBM(i,True)
            SF = self.proj.GetNodeShear(i,True)
            self.df_disp.loc[i,'Disp'] =i_disp
            self.df_disp.loc[i,'BM'] =BM
            self.df_disp.loc[i,'SF'] =SF
            self.df_disp.loc[i,'Level'] =self.proj.GetNodeLevel(i)
        return self.df_disp
        
    def new_file(self, filename,JN='',initial='', title='',subtitle='',calc_heading='',Notes=''):
        self.proj.NewFile(filename)
        self.proj.SetJobNumber(JN)
        self.proj.SetInitials(initial)
        self.proj.SetJobTitle(title)
        self.proj.setSubTitle(subtitle)
        self.proj.setNotes(Notes)
        logger.debug(f"Started New file {filename}")
    def save(self):
        self.proj.Save()
        logger.debug('File saved successfully')
        
    def set_section(self, section_id, section_string, input_type, effective_width, EI):
        self.proj.SetSection(section_id, section_string, input_type, effective_width, EI)

    def set_node_load_displacement(self, node_ID, force, moment, displacement):
        '''
        Adds an applied load and/or displacement at the node 
        @param:
        node_ID: 
        force: shear force at the node
        moment: bending moement at the node
        displacement: displacement at the node
        '''

        self.proj.SetNodeLoadDisp(node_ID, force, moment, displacement)

    def analyse(self,verbose=False):
        '''
        Run the calculation 
        '''
        self.proj.Analyse()
        if verbose:
            self.summary()

    def get_node_BM(self,node_ID, is_below=True):
        '''
        Return the bending moement at a node 
        @param:

        node_ID:
        is_below:`True` or `False`
        '''
        return self.proj.GetNodeBM(node_ID, is_below)
    def plot_force_Diplacement(self):
        '''
        plot the displacement and bending moment profile along the pile
        return: plotly figure object
        '''

        df = self.get_nodal_displacement_BM()

        trace1 = go.Scatter(x=df['Disp'], y=df['Level'], name='Displacement (m)')
        trace2 = go.Scatter(x=df['BM'], y=df['Level'], name='Bending Moment (kNm)')
        trace3 = go.Scatter(x=df['SF'], y=df['Level'], name='Shear Force (kN)')

        # Create the subplots
        xtitles = ['Displacement (mm)', 'Bending Moment (kNm)', 'Shear Force (kN)']
        fig = make_subplots(rows=1, cols=3)

        # Add the traces to the subplots
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=2)
        fig.add_trace(trace3, row=1, col=3)

        # Update the layout of each subplot
        for i in range(3):
            fig.update_xaxes(title=xtitles[i], showgrid=True, gridcolor='lightgray', side='top',row=1, col=i+1)
            fig.update_yaxes(title='Level', showgrid=True, gridcolor='lightgray', side='top',row=1, col=i+1)

        # Update the layout of the figure
        fig.update_layout(
            width=1000,
            height=800,
            template='simple_white',
        )
        # Show the plot
        return fig
    def max_bending(self):
        return self.proj.MaxBM()
    def max_shear(self):
        return self.proj.MaxShear()
    def max_displacement(self):
        return self.proj.MaxDisp()

    def summary(self):
        summary_string = f'''
Max Bending Moment:{self.max_bending():10,.2f}kNm,
Max Shear Force   :{self.max_shear():10,.2f}kN,
Max Displacement  :{self.max_displacement():10,.2f}mm
        '''
        print(summary_string)

