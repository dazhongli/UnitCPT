from .cpt import CPT
from pathlib import Path
import os
import plotly.io as pio
import numpy as np
import pandas as pd
from numpy import degrees, log10, pi, radians, tan, sin, cos
import matplotlib.pyplot as plt

def calc_su(qt, sigma_v, nkt):
    '''
    This function calculates the undrained shear strength at depth z for clay
    '''
    return (qt * 1000 - sigma_v) / nkt

def calc_su1(su, su0, z):
    '''
    This function calculates the rate of increase of shear strength with depth in linearly increasing strength profiles, as
determined from DSS testing for clay
    '''
    if su <= su0:
        return 0.001 / z
    else: 
        return (su - su0) / z

def calc_alpha(su, sigma_v_e):
    '''
    This function calculates the dimensionless skin friction factor alpha
    Only applys for clay
    '''
    psi = su / sigma_v_e
    if psi > 1.0:
        alpha =0.5 * psi ** (-0.25)
    else:
        alpha =0.5 * psi ** (-0.5)
    return alpha

def calc_N_pd(alpha):
    '''
    This function calculates the lateral bearing capacity factor for flow around mechanism N_pd for clay
    '''
    return 9 + 3 * alpha

def calc_d(su0, su1, D):
    '''
    This function calculates the model parameter d for clay
    '''
    lam = su0 / (su1 * D)
    return max(16.8 - 2.3 * log10(lam), 14.5)

def calc_N_p0(N_1, N_2, alpha, d, D, N_pd, z):
    '''
    This function calculates the lateral bearing capacity factor due to passive wedge for weightless soil N_p0 for clay
    '''
    #N_p0 = N_1 - (1 - alpha) - (N_1 - N_2) * (1 - (z / (d * D))**0.6)**1.35
    k = 1 - (z / (d * D))**0.6
    if k>0:
        N_p0 = N_1 - (1 - alpha) - (N_1 - N_2) * k**1.35
    else:
        N_p0 = N_1 - (1 - alpha)
    return min(N_p0, N_pd)

def calc_N_P(N_pd, N_p0, gamma, z, su, isotropy):
    '''
    This function calculates the total lateral bearing capacity factor N_P for clay
    '''
    if z > 10:
        N_P = N_pd
    else:
        if su < 15:
            N_P = min(N_p0 + ((gamma - 10) * z) / su, N_pd)
        else:
            N_P = min(2 * N_p0, N_pd)

    if z == 0:
        N_P0 = N_P

    if isotropy != 'true':
        C_W = 1 + (0.9 - 1) * (N_pd - N_P) / (N_pd - N_P0)
        N_P = min(C_W * N_p0 + ((gamma - 10) * z) / su, N_pd)
    
    return N_P

def calc_pu(su, D, N_P):
    '''
    This function calculates the ultimate soil resistance per unit length (in units of force per unit length) pu for clay
    '''
    return su * D * N_P

def calc_OCR(Qt1):
    '''
    This function calculates the OCR for clay
    '''
    return 0.25 * Qt1 ** 1.25

def calc_p_mo(pu, id_p):
    '''
    This function calculates the p_mo for clay
    '''
    index = [0.000, 0.050, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 0.975, 1.000]
    p_mo  = [x * pu for x in index]
    return p_mo[id_p]

def calc_y_mo(I_p, OCR, D, id_p):
    '''
    This function calculates the y_mo for clay
    '''
    # define the table as a pandas DataFrame
    #index = [0.000, 0.050, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 0.975, 1.000]
    table_I_p_above_30 = pd.DataFrame({
        'OCR ≤ 2': [0.0000, 0.0003, 0.0030, 0.0053, 0.0090, 0.0140, 0.0220, 0.0320, 0.0500, 0.0820, 0.1500, 0.2500, float('inf')],
        'OCR = 4': [0.0000, 0.0004, 0.0040, 0.0080, 0.0150, 0.0240, 0.0360, 0.0550, 0.0840, 0.1400, 0.2300, 0.3000, float('inf')],
        'OCR = 10': [0.0000, 0.0005, 0.0050, 0.0110, 0.0210, 0.0340, 0.0520, 0.0780, 0.1200, 0.1900, 0.3000, 0.4000, float('inf')]
    })

    table_I_p_below_30 = pd.DataFrame({
        'OCR ≤ 2': [0.0000, 0.0001, 0.0010, 0.0018, 0.0030, 0.0048, 0.0073, 0.0110, 0.0170, 0.0270, 0.0500, 0.0830, float('inf')],
        'OCR = 4': [0.0000, 0.0002, 0.0020, 0.0040, 0.0075, 0.0120, 0.0180, 0.0270, 0.0420, 0.0700, 0.1100, 0.1500, float('inf')],
        'OCR = 10': [0.0000, 0.0003, 0.0033, 0.0073, 0.0140, 0.0230, 0.0350, 0.0520, 0.0800, 0.1300, 0.2000, 0.2700, float('inf')]
    })

    # determine the row to use based on the Ip value
    if I_p > 30:
        table = table_I_p_above_30
    else:
        table = table_I_p_below_30

    # determine the column to use based on the OCR value
    y_results = []
    if OCR <= 2:
        y_results = table['OCR ≤ 2']
    elif OCR > 2 and OCR <=4:
        for i in range(table.iloc[:,1].size):
        # interpolate the ymo/D value for the given OCR value
            y_value = np.interp(OCR, [2, 4], [table.iloc[i]['OCR ≤ 2'], table.iloc[i]['OCR = 4']])
            y_results.append(y_value)
    elif OCR > 4 and OCR <10:
        for i in range(table.iloc[:,0].size):
            y_value = np.interp(OCR, [4, 10], [table.iloc[i]['OCR = 4'], table.iloc[i]['OCR = 10']])
            y_results.append(y_value)
    else:
        y_results = table['OCR = 10']
    
    y_mo  = [x * D for x in y_results]

    return y_mo[id_p] * 1000

def identify_soil_layers(df):
    # Initialize variables
    prev_soil_type = None
    layer_counts = []
    layer_thicknesses = []
    layer_types = []
    layer_strengths_0 = []

    # Iterate over rows in the DataFrame
    for i, row in df.iterrows():
        # Check if this row has a different soil type than the previous row
        if row['soil_type'] != prev_soil_type:
            # If this is not the first layer, record the count, thickness, soil type, and initial strength of the previous layer
            if prev_soil_type is not None:
                layer_counts.append(layer_count)
                layer_thickness = df.loc[i-1, 'SCPT_DPTH'] - df.loc[i-layer_count, 'SCPT_DPTH']
                layer_thicknesses.append(layer_thickness)
                layer_types.append(prev_soil_type)
                layer_strengths_0.append(df.loc[i-layer_count, 'su'])
            # Reset the layer count and update the previous soil type
            layer_count = 1
            prev_soil_type = row['soil_type']
        else:
            # Increment the layer count if the soil type is the same as the previous row
            layer_count += 1
    
    # Record the count, thickness, soil type, and initial strength of the last layer
    layer_counts.append(layer_count)
    layer_thickness = df['SCPT_DPTH'].max() - df.loc[len(df)-layer_count, 'SCPT_DPTH']
    layer_thicknesses.append(layer_thickness)
    layer_types.append(prev_soil_type)
    layer_strengths_0.append(df.loc[len(df)-layer_count, 'su'])

    # Add the 'strength_0' column to the DataFrame
    df['su0'] = pd.Series([val for val, count in zip(layer_strengths_0, layer_counts) for i in range(count)], index=df.index)

    # Append the 'strength_0' value to each row of the DataFrame
    for i, row in df.iterrows():
        row['su0'] = layer_strengths_0[layer_types.index(row['soil_type'])]


def calc_h_f(p_mo, p_u, z, D):
    '''
    This function calculates the hybrid factor h_f at each depth and for all points p_mo/p_u on normalized monotonic p-y curves for clay
    '''
    z_rot = 15 * D
    if z <= z_rot:
        h_f = p_mo / p_u - (z / z_rot)**2
    else:
        h_f = p_mo / p_u - 1
    h_f = max(h_f, -1)
    return min(h_f, 0.99)


def calc_N_eq(h_f, clay_type):
    '''
    This function calculates the number of equivalent cycles N_eq at each depth and for all points p_mo/p_u on normalized monotonic p-y curves for clay
    '''
    if clay_type == 'Gulf of Mexico':
        g = 1.0
    elif clay_type == 'North Sea soft clay':
        g = 1.25
    elif clay_type == 'North Sea stiff clay':
        g = 2.5
    return min((2 / (1 - h_f)) ** g, 25)

def calc_p_y_mod(N_eq, clay_type):
    '''
    This function calculates the p-modifier and the y-modifier at each depth and for all points p_mo/p_u on normalized monotonic p-y curves for clay
    '''
    if clay_type == 'Gulf of Mexico':
        p_mod = 1.47 - 0.14 * np.log(N_eq)
        y_mod = 1.2 - 0.14 * np.log(N_eq)
    elif clay_type == 'North Sea soft clay':
        p_mod = 1.63 - 0.15 * np.log(N_eq)
        y_mod = 1.2 - 0.17 * np.log(N_eq)
    elif clay_type == 'North Sea stiff clay':
        p_mod = 1.45 - 0.17 * np.log(N_eq)
        y_mod = 1.2 - 0.17 * np.log(N_eq)
    return p_mod, y_mod

def plot_p_y_cyclic(df, plot_interval):
    '''
    Plot the p-y curve for sand at different depths with specified rows to plot
    For example, rows_to_plot = [200, 800, 1600, 3200, 5600, 6400]
    '''

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # get the index of rows with the given interval
    index_list = df.iloc[::plot_interval].index.tolist()

    # Plot the selected rows
    for i in index_list:
        row = df.iloc[i]
        if row['soil_type'] == 'clay':
            x = [row[f'y_cy{j}'] for j in range(11)]
            y = [row[f'p_cy{j}'] for j in range(11)]           
            depth = round(row['SCPT_DPTH'], 2)
            ax.plot(x, y, '-o', label=f'Depth_cyclic: {depth}m')

            x_mo = [row[f'y{j}'] for j in range(11)]
            y_mo = [row[f'p{j}'] for j in range(11)]
            ax.plot(x_mo, y_mo, '-o', label=f'Depth_mono: {depth}m')

    # Add labels and legend
    ax.set_xlabel('y (mm)')
    ax.set_ylabel('p (kN/m)')
    ax.legend()

    # Show the plot
    plt.show()
    return fig