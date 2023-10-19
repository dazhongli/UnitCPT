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

def calc_N_p0(N1, N2, alpha, d, D, N_pd, z):
    '''
    This function calculates the lateral bearing capacity factor due to passive wedge for weightless soil N_p0 for clay
    '''
    N_p0 = N1 - (1 - alpha) - (N1 - N2) * (1 - (z / (d * D))**0.6)**1.35
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
        for i in table.iloc[:,0].size:
            y_value = np.interp(OCR, [4, 10], [table.iloc[i]['OCR = 4'], table.iloc[i]['OCR = 10']])
            y_results.append(y_value)
    else:
        y_results = table['OCR = 10']
    
    y_mo  = [x * D for x in y_results]

    return y_mo[id_p]
