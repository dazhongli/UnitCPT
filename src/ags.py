import pandas as pd
import re
import csv


class AGSParser:
    def __init__(self, ags_format=1):
        '''
        @param:
        ags_format: define the format of the ags file
        '''
        self.ags_format = ags_format

    def read_ags_file(self, filename):
        with open(filename, 'r') as fin:
            ags_str = fin.read()
        # Get the keys within the ags file
        self.ags_str = ags_str
        self.keys = AGSKeys(re.findall('"GROUP","(\w+)"', ags_str))
        self.key_IDs = re.findall('"GROUP","(\w+)"', ags_str)
        if self.ags_format == 1:
            self._search_key = r'"\*{2}\??key"(.*)\n([\s\S]*?)(?:\n{2,}|\Z)'
        else:
            self._search_key = r'"GROUP","key"\n([\s\S]*?)(?:\n{2,}|\Z)'
        try:
            df_hole = self.get_df_from_key('HOLE')
            hole_list = list(set(df_hole.drop(0, axis=0).HOLE_ID))
            self.holes = HoleKeys(hole_list)
        except Exception as e:
            print(e)
            Warning('No Hole ID found!')

    def extract_str_block(self, key=''):
        '''
        Return the string blocks given a kepy
        '''
        search_key = self._search_key.replace('key', key)
        if self.ags_format == 1:
            block = re.findall(search_key, self.ags_str)[0][1]
        else:
            block = re.findall(search_key, self.ags_str)[0]

        return block

    def get_df_from_key(self, key='', hole_id=''):
        s = self.extract_str_block(key)
        self.active_df = self._parse_data_to_df(s)
        self.active_key = key
        if hole_id is not '':
            return self.active_df[self.active_df['HOLE_ID'] == hole_id]
        else:
            return self.active_df

    def _parse_data_to_df(self, s):
        lines = s.split('\n')
        data = []
        for j in range(len(lines)):
            line = lines[j]
            while line.endswith(','):  # if the line ends with ',', it has not finished
                line = line + lines[j+1]
                j = j + 1
            line_info = re.findall('"(.*?)"', line)
            if r'<CONT>' in line:
                for i in range(1, len(line_info)):
                    data[-1][i] = data[-1][i] + line_info[i]
                continue
            data.append(re.findall('"(.*?)"', line))
        column = [x.replace('*', '') for x in data[0]]
        df = pd.DataFrame(data[1:])
        df.columns = column
        return df

    def get_key_group(self, key):
        data_str = self.extract_str_block(key)
        df = self._parse_data_to_df(data_str)
        return df


class AGSKeys:
    def __init__(self, keys):
        for key in keys:
            setattr(self, key, key)
        self.key_map = {
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
            'SAMP': 'Sample Information'
        }


class HoleKeys:
    def __init__(self, keys):
        for key in keys:
            setattr(self, key, key)
