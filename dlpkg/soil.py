import numpy as np
from numpy.core.fromnumeric import cumsum
import pandas as pd
from scipy.interpolate import interp1d
# import utilities as utl


class Stratum:
    def __init__(self, **kwargs):
        '''
        required input:
        bottom_level: an array that defines the bottom levels of layer
        gamma: the submerged unit weight of the soils, in kn/m3
        drainage: UD - undrained, Dr- drained
        name: string, which will show the name of the soil
        '''
        self.keywords = kwargs
        self.df_soil = pd.DataFrame(columns=['bottom_level',
                                             'gamma',  # use bulk density
                                             'name',
                                             'drainage'])
        if kwargs is not None:
            for key in kwargs:
                self.df_soil[key] = kwargs[key]
        self.refresh()

    def refresh(self):
        kwargs = self.keywords
        if 'water_table' not in kwargs:
            self.water_table = 0  # by defult water table is at the ground surface
            self.df_soil['u_static'] = (self.df_soil.bottom_level*10)
        # if the water table is below the ground level, we need to insert a level:
        else:
            self.water_table = kwargs.get('water_table')
            self.df_soil = utl.interp_row(
                self.df_soil, val=self.water_table, by_col='bottom_level')
            self.df_soil['u_static'] = self.df_soil.bottom_level.apply(
                lambda x: 0 if x <= self.water_table else (x-self.water_table)*10)

        if ('bottom_level' in kwargs) and ('gamma' in kwargs):
            self.df_soil['thickness'] = (self.df_soil.bottom_level -
                                         self.df_soil.bottom_level.shift(1)).fillna(self.df_soil.bottom_level[0])
            self.df_soil['sigma_v_e'] = (
                self.df_soil.gamma * self.df_soil.thickness).cumsum() - self.df_soil['u_static']
            x_values = self.df_soil.bottom_level.values
            y_values = self.df_soil.sigma_v_e.values
            # let's insert 0 in front of them
            x_values = np.insert(x_values, 0, 0)
            y_values = np.insert(y_values, 0, 0)
            self._effective_stress = interp1d(
                x=x_values, y=y_values)

    def effective_stress(self, level):
        '''
        Return the stress level at this point
        '''
        try:
            return self._effective_stress(level)
        except:
            return np.nan

    def add_soil_names_df(self, df_input, depth_key):
        '''
        This function interpolate the soil layers and add the soil names to teh dataframe
        df_input must hold the data with the index defines that soil depth
        '''
        df_temp = df_input.set_index(depth_key)
        insert_list = [
            x for x in self.df_soil.bottom_level if x < df_temp.index.max()]
        for level in insert_list:
            df_temp.loc[level, :] = np.nan
        df_temp = df_temp.sort_index()
        df_temp = df_temp.interpolate('linear')
        df_temp = df_temp.reset_index()

        def f_name(x): return self.df_soil.loc[np.where(
            self.df_soil.bottom_level >= x)[0][0], 'name']

        def f_drainage(x): return self.df_soil.loc[np.where(
            self.df_soil.bottom_level >= x)[0][0], 'drainage']
        df_temp['name'] = df_temp[depth_key].map(f_name)
        df_temp['drainage'] = df_temp[depth_key].map(f_drainage)
        return df_temp

    def assgin_su(self, su_func, soil_name, relative=True):
        '''
        This function assign the undrained shear strength to soil layers by a soil name
        su_func: strength profile of soil layer as a function of the depth, either absolute or relative
        '''
        df = self.df_soil
        key = 'thickness' if relative else 'bottom_level'
        if 'su' in df.columns:
            df.su = df.su + df.apply(lambda row: su_func(row[key])
                                     if row['name'] == soil_name else 0, axis=1)
        else:
            df['su'] = df.apply(lambda row: su_func(row[key])
                                if row['name'] == soil_name else 0, axis=1)
        return self.df_soil
