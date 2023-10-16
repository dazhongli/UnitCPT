from dlpkg.cpt import CPT
from pathlib import Path
import os
import plotly.io as pio
import numpy as np
import pandas as pd
from numpy import degrees, log10, pi, radians, tan, sin, cos

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

def calc_A(D, z, loading):
    '''
    This function calculates the factor to account for static or cyclic actions, for p-y curves for sand
    '''
    if loading == 'Monotonic':
        return max((3.0 - 0.8 * z / D), 0.9)
    else:
        return 0.9
    
def calc_p(y, A, pr, z, k):
    '''
    This function calculates the lateral soil resistance-displacement p-y relationship for a pile in sand
    '''
    return A * pr * np.tanh(k * z / (A * pr) * y / 1000)


