#from .cpt import CPT
from pathlib import Path
import os
import plotly.io as pio
import numpy as np
import pandas as pd
from numpy import degrees, log10, pi, radians, tan, sin, cos
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.interpolate import interp1d
from enum import Enum

class Loading_type(Enum):
    Cyclic = 1
    Monotonic = 2

def determine_soil_type(ic):
    '''
    This function determines soil type based on 'Ic' value
    '''
    if pd.isna(ic):
        return 'N/A'
    elif ic > 2.6:
        return 'clay'
    else:
        return 'sand'

def calc_phi_e(qt, sigma_v, sigma_v_e):
    '''
    This function calculates the effective friction angle based on excel spreedsheet
    Only applys for sand
    '''
    return 17.6 + 11 * log10((qt * 1000 - sigma_v) / sigma_v_e)

def calc_C1(phi_e):
    '''
    This function calculates the dimensionless coefficient C1 determined as function of ϕ′, for p-y curves for sand
    '''
    #phi_e = calc_phi_e()
    alpha = phi_e / 2
    beta = 45 + phi_e / 2
    K0 = 0.4
    phi_e = radians(phi_e)
    alpha = radians(alpha)
    beta = radians(beta)
    return ((tan(beta))**2 * tan(alpha)) / (tan(beta - phi_e)) + K0 * (tan(phi_e) * sin(beta) / (cos(alpha) * tan(beta - phi_e)) + tan(beta) * (tan(phi_e) * sin(beta) - tan(alpha)))

def calc_C2(phi_e):
    '''
    This function calculates the dimensionless coefficient C2 determined as function of ϕ′, for p-y curves for sand
    '''
    #phi_e = self.calc_phi_e()
    beta = 45 + phi_e / 2
    phi_e = radians(phi_e)
    beta = radians(beta)
    Ka = (1 - sin(phi_e)) / (1 + sin(phi_e))
    return tan(beta) / tan(beta - phi_e) - Ka

def calc_C3(phi_e):
    '''
    This function calculates the dimensionless coefficient C3 determined as function of ϕ′, for p-y curves for sand
    '''
    #phi_e = self.calc_phi_e()
    beta = 45 + phi_e / 2
    phi_e = radians(phi_e)
    beta = radians(beta)
    Ka = (1 - sin(phi_e)) / (1 + sin(phi_e))
    K0 = 0.4
    return Ka * (tan(beta)**8 - 1) + K0 * tan(phi_e) * tan(beta)**4

def calc_k(phi_e):
    '''
    This function calculates the initial modulus of subgrade reaction, for p-y curves for sand
    '''
    #phi_e = self.calc_phi_e()

    # Define the values of phi and k for the table
    phi_table = np.array([25, 30, 35, 40])
    k_table = np.array([5400, 8700, 22000, 45000])

    # Find the indices of the two nearest values of phi in the table
    idx = np.searchsorted(phi_table, phi_e)
    if idx == 0:
        # phi is less than all values in the table
        return k_table[0]
    elif idx == len(phi_table):
        # phi is greater than all values in the table
        return k_table[-1]
    else:
        # Interpolate k using linear interpolation
        phi1 = phi_table[idx-1]
        phi2 = phi_table[idx]
        k1 = k_table[idx-1]
        k2 = k_table[idx]
        return k1 + (k2 - k1) * (phi_e - phi1) / (phi2 - phi1)

def calc_pr(D, gamma, z, C1, C2, C3):
    '''
    This function calculates the representative lateral capacity (kN/m), for p-y curves for sand
    '''
    #phi_e = self.calc_phi_e()
    gamma_e = gamma - 10
    p_rs = (C1 * z + C2 * D) * gamma_e * z
    p_rd = C3 * D * gamma_e * z
    return min(p_rs, p_rd)

def calc_A(D, z, monotonic):
    '''
    This function calculates the factor to account for static or cyclic actions, for p-y curves for sand
    '''
    if monotonic == True:
        return max((3.0 - 0.8 * z / D), 0.9)
    else:
        return 0.9
    
def calc_p(y, A, pr, z, k):
    '''
    This function calculates the lateral soil resistance-displacement p-y relationship for a pile in sand
    '''
    return A * pr * np.tanh(k * z / (A * pr) * y / 1000)

def calc_y(y_range, i):
    '''
    This function calculates the lateral soil resistance-displacement p-y relationship for a pile in sand
    '''
    logspace_points = [0]
    logspace_points += list(np.logspace(0, np.log10(y_range), 11, base=10.0, endpoint=True))
    return logspace_points[i]

def export_p_y_monotonic(df, filename):
    '''
    Export monotonic p-y data to excel
    '''
    df_export = pd.DataFrame()
    df_export ['Depth'] = df.loc[:, ['SCPT_DPTH']]
    for i in range(12):
        df_export [f'y{i}'] = df.loc[:, [f'y{i}']]
        df_export [f'p{i}'] = df.loc[:, [f'p{i}']]
    df_export.to_excel(filename, index=False)
    print(f"{filename} has been exported successfully.")
    #writer = pd.ExcelWriter('data.xlsx')
    #df.to_excel(writer, sheet_name='Sheet 1')
    #writer.save()

def export_p_y_cyclic(df, filename):
    '''
    Export cyclic p-y data to excel
    '''
    df_export = pd.DataFrame()
    df_export ['Depth'] = df.loc[:, ['SCPT_DPTH']]
    for i in range(12):
        df_export[f'y{i}'] = df.apply(lambda row: row[f'y_cy{i}'] if row['soil_type'] == 'clay' else row[f'y{i}'], axis=1)
        df_export[f'p{i}'] = df.apply(lambda row: row[f'p_cy{i}'] if row['soil_type'] == 'clay' else row[f'p{i}'], axis=1)
        #if df.loc[df['soil_type'] == 'clay']:
            #df_export [f'y_cy{i}'] = df.loc[:, [f'y_cy{i}']]
            #df_export [f'p_cy{i}'] = df.loc[:, [f'p_cy{i}']]
        #else:
            #df_export [f'y{i}'] = df.loc[:, [f'y{i}']]
            #df_export [f'p{i}'] = df.loc[:, [f'p{i}']]
    df_export.to_excel(filename, index=False)
    print(f"{filename} has been exported successfully.")

def plot_p_y_curve(df, plot_interval):
    '''
    Plot the p-y curve for sand at different depths with specified rows to plot
    For example, rows_to_plot = [200, 800, 1600, 3200, 5600, 6400]
    '''

    # Create a figure object
    fig = go.Figure()

    # get the index of rows with the given interval
    index_list = df.iloc[::plot_interval].index.tolist()

    max_x = float('-inf')
    max_y = float('-inf')

    # Plot the selected rows
    for i in index_list:
        row = df.iloc[i]
        x = [row[f'y{j}'] for j in range(12)]
        y = [row[f'p{j}'] for j in range(12)]
        depth = round(row['SCPT_DPTH'], 2)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'{depth}m'))

        # Update the maximum x and y values if necessary
        if max(x) > max_x:
            max_x = max(x)
        if max(y) > max_y:
            max_y = max(y)

    # Add labels and legend
    fig.update_layout(xaxis_title='y (mm)', yaxis_title='p (kN/m)', legend=dict(title='Depth'), plot_bgcolor='white')
    fig.update_xaxes(range=[0, max_x*1.05], showgrid=True, gridcolor='gainsboro', linecolor='black', tickcolor='black', ticks="outside")
    fig.update_yaxes(range=[0, max_y*1.05], showgrid=True, gridcolor='gainsboro', linecolor='black', tickcolor='black', ticks="outside")

    # Show the plot
    fig.show()
    #return fig

def interpolate_cpt_data(df, interval):
    new_depths = pd.Series(np.arange(int(df['SCPT_DPTH'].min())+interval, int(df['SCPT_DPTH'].max()), interval))
    col_names = df.iloc[:, 2:].columns.tolist()
    df_resampled = pd.DataFrame()
    df_resampled['SCPT_DPTH'] = new_depths
    for col_name in col_names:
        func_name = interp1d(x = df['SCPT_DPTH'], y = df[col_name], kind = 'linear')
        df_resampled[col_name] = func_name(new_depths)
    return df_resampled

'''
def shaft_friction_unified_sand(pile, z, qc, sigma_v_e, compression):

    R_star = np.sqrt((pile.dia_out/2)**2 - (pile.dia_inner/2)**2)
    delta = radians(29)
    Ar = pile.disp_ratio
    D = pile.dia_out
    h= pile.penetration-z
    depth_ratio = h/D
    qc = qc*1000  # qc in MPa

    sigma_rc = qc / 44 * Ar ** (0.3) * np.max([depth_ratio, 1])** (-0.4)
    delta_sigma = qc / 10 * (qc / sigma_v_e) ** (-0.33) * 0.0356 / D

    if compression == 'true':
        fl = 0.75
    else:
        fl = 0.75

    return fl * (sigma_rc + delta_sigma) * tan(delta)

def base_Qb_unified_sand(pile, qp_average):
    Ar = pile.disp_ratio
    qp_average = 1000*qp_average
    return (0.12 + 0.38 * Ar) * qp_average * pile.gross_area

'''