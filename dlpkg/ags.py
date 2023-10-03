import csv
import re
from enum import Enum
import pandas as pd
import numpy as np


class AGSFormat(Enum):
    AGS4 = 1
    AGS3 = 2
    AGS31 = 3


AGS3_key_map = {
    'WETH': 'Weathering information',
    'PROJ': 'Project Information',
    'FLSH': 'Drilling Flush Details',
    'CDIA': 'Casing Diameter by Depth',
    'CORE': 'Coring Information',
    'DETL': 'Stratum Detail Descriptions',
    'FRAC': ' Fracture Spacing ',
    'GEOL': 'Field Geological Descriptions',
    'HDIA': 'Hole Diameter by Depth ',
    'ISPT': 'Standard Penetration Test Results ',
    'IVAN': 'In Situ Vane Tests',
    'PTIM': 'Boring/Drilling Progress by Time',
    'SAMP': 'Sample Information',
    'IDEN': 'Insitu Density Test',
    'POBS': 'Piezometer Readings',
    'PREF': 'Piezometer Installation Details',
    'IPRM': 'In Situ Permeability Test'
}


class AGSParser:
    def __init__(self, ags_str):
        self.ags_str = ags_str

        # Let's check the format of the ags file here

        if '"GROUP","PROJ"' in ags_str:
            ags_format = AGSFormat.AGS4
        elif '**PROJ**' in ags_str:
            ags_format = AGSFormat.AGS3
        else:
            ags_format = AGSFormat.AGS31
        print(f'AGS_format = {ags_format}')
        self.key_IDs = re.findall('"GROUP","(\w+)"', ags_str)
        self.ags_format = ags_format
        if self.ags_format == AGSFormat.AGS3:
            self._keys = AGSKeys(re.findall(r"\*\*\??(\w+)", ags_str))
            self._search_key = r'(?s)\*\*key"\n(.+?)"\n"\*\*|$'
        elif self.ags_format ==AGSFormat.AGS31: # AGS3.1
            self._keys = AGSKeys(re.findall(r"\*\*\??(\w+)", ags_str))
            self._search_key = r'(?s)\*\*key"\n(.+?)"\*\*|$'
        else: # AGS4
            self._keys = AGSKeys(re.findall('"GROUP","(\w+)"', ags_str))
            self._search_key = r'"GROUP","key"\n([\s\S]*?)(?:\n{2,}|\Z)'
        try:
            df_hole = self._get_df_from_key('HOLE')
            hole_list = list(set(df_hole.drop(0, axis=0).HOLE_ID))
            self.holes = HoleKeys(hole_list)
        except Exception as e:
            print(e)
            Warning('No Hole ID found!')

    @property
    def groups(self):
        return self._keys.__dict__

    def extract_str_block(self, key=''):
        '''
        Return the string blocks given a key
        '''
        search_key = self._search_key.replace('key', key)
        if self.ags_format == AGSFormat.AGS3:
            try:
                block = re.findall(search_key, self.ags_str)[0]
            except:
                print(f'{search_key}--pattern NOT FOUND!')
                # raise(f'{search_key} --Pattern NOT FOUND!')
        else:
            block = re.findall(search_key, self.ags_str)[0]
        return block

    def group_df(self, group_name, hole_id=''):
        '''
        return the data of a group in a format of dataframe 
        '''
        if group_name == 'HOLE':
            return self._get_df_from_key('HOLE', hole_id=hole_id)
        if 'HOLE' in self.groups.keys():
            df_hole = self._get_df_from_key('HOLE', hole_id=hole_id)
            df_group = self._get_df_from_key(group_name, hole_id=hole_id)
            return pd.merge(df_hole[['HOLE_ID', 'HOLE_GL', 'HOLE_NATE', 'HOLE_NATN','HOLE_INCL']], df_group, on='HOLE_ID')

    def _get_df_from_key(self, key='', hole_id=''):
        '''
        Extract the group data and return a data frame
        '''
        s = self.extract_str_block(key)
        df = self._parse_data_to_df(s)
        df = df.replace(['', ' '], np.nan)
        # remove the columns with all data are null
        df = df.loc[:, ~(df.iloc[1:].isnull().all())]
        if hole_id != '':
            return df[df['HOLE_ID'] == hole_id]
        else:
            return df

    def _parse_data_to_df(self, s):
        lines = s.split('\n')
        data = []
        j = 0
        while j < len(lines)-1:
            line = lines[j]
            while line.endswith(','):  # if the line ends with ',', it has not finished
                line = line + lines[j+1]
                j = j + 1
                continue
            line_info = re.findall('"(.*?)"', line)
            if r'<CONT>' in line:
                for i in range(1, len(line_info)):
                    data[-1][i] = data[-1][i] + line_info[i]
                j = j + 1
                continue
            data_line = (re.findall('"(.*?)"', line))
            # insert_line = [float(x) if x.isnumeric() or (x[0] == '-' and x[1:].isnumeric()) else x for x in data_line]
            insert_line = data_line
            
            data.append(insert_line)
            j = j + 1 
        column = [x.replace('*', '') for x in data[0]]
        df = pd.DataFrame(data[1:])
        df.columns = column
        return df

    def get_key_group(self, key):
        data_str = self.extract_str_block(key)
        df = self._parse_data_to_df(data_str)
        return df

    def __str__(self):
        if self.ags_format == AGSFormat.AGS3:
            ags_version = 3
        print_string = f'''
        AGS_version = {ags_version}
        Number of BH = '''


class AGSKeys:
    def __init__(self, keys):
        for key in keys:
            if key in AGS3_key_map.keys():
                setattr(self, key, AGS3_key_map[key])
            else:
                setattr(self, key, key)


class HoleKeys:
    def __init__(self, keys):
        for key in keys:
            setattr(self, key, key)
