from cpt import CPT
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
    This function calculates the factor N_pd
    Only applys for clay
    '''
    return 9 + 3 * alpha

def calc_d(su0, su1, D):
    '''
    This function calculates the model parameter for clay
    '''
    lam = su0 / (su1 * D)
    return max(16.8 - 2.3 * log10(lam), 14.5)