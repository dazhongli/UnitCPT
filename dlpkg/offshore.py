import pandas as pd
from scipy.integrate import quad
import numpy as np
from numpy import pi, cos, cosh, sin, sinh, tanh
import plotly.graph_objects as go
import logging as log
logger = log.getLogger()
logger.setLevel(log.DEBUG)


class WaveForce():
    def __init__(self, Ci, Cd, D):
        '''
        Ci = Inertia Coefficient
        Cd = Drag Coefficient
        D  = Diameter of Pile
        '''
        self.Ci = Ci
        self.Cd = Cd
        self.wave = Wave(Hmax=np.nan, Hs=np.nan, density=1.025,
                         L=np.nan, T=np.nan, Hd=np.nan, d=np.nan)
        self.D = D

    def wave_function(self, x, xc, H, L):
        '''
        The wave function, return the height of the free surface, Z as in Figure 18 of the portworks manual
        '''
        return H/2 * cos(2*pi*(x-xc)/L)

    def h_velocity(self, z, d, x, t, type='linear'):
        '''
        Calculate the horizontal particle velocity
        '''
        k = self.wave.k
        w = self.wave.omega
        H = self.wave.Hd
        c = self.wave.c

        if type == 'linear':
            c1 = cosh(k*(z+d))
            c2 = cosh(k*d)
            g = 9.81
            return (g*k*H/(2*w) * (c1/c2) * cos(k*(x-c*t)))
        else:
            raise('Not implemented')

    def h_acceleration(self, z, d, x, t, type='linear'):
        '''
        Calculate the horizontal particle velocity
        '''
        k = self.wave.k
        H = self.wave.Hd
        c = self.wave.c

        if type == 'linear':
            c1 = cosh(k*(z+d))
            c2 = cosh(k*d)
            g = 9.81
            return (g*k*H/(2) * (c1/c2) * sin(k*(x-c*t)))
        else:
            raise('Not implemented')

    def F_inertia(self, z, x0, xc, max=False):
        '''
        Calculate the inertia force

        '''
        Ci = self.Ci
        rou = self.wave.density
        D = self.D
        d = self.wave.d
        t = xc/self.wave.c

        # the inertial force will have the same direction as the acceleration
        return (Ci * rou*pi*D**2)/4*self.h_acceleration(z, d, x0, t)

    def F_drag(self, z, x0, xc):
        Cd = self.Cd
        rou = self.wave.density
        D = self.D
        d = self.wave.d
        t = xc/self.wave.c
        u = self.h_velocity(z, d, x0, t)
        return 1/2 * Cd * rou * D * abs(u) * u

    def orbit_width_of_water_particles(self, H_design, dw, L):
        '''
        Calculate teh orbit width of water particles
        H_design: design wave height
        dw:       design water height
        L:        design wave length
        '''
        return H_design/tanh(2*pi*dw/L)


class Wave():
    '''
    This class will handle all the calculation and all the consersion related to the waves
    '''

    def __init__(self, Hmax, Hs, Hd, L, T, density, d, c=1.0):
        self.Hmax = Hmax
        self.Hs = Hs
        self.Hd = Hd  # design wave height
        self.d = d
        self.L = L
        self.T = T
        self.density = density
        self.c = c

    def set_vals(self, **kwarg):
        for key in kwarg:
            self.__setattr__(key, kwarg[key])
        if 'T' in kwarg:
            self.omega = 2*pi/self.T
        if 'L' in kwarg:
            self.k = 2*pi/self.L

    def wave_profile(self, x, t):
        '''
        '''
        H = self.Hd
        k = self.k
        c = self.c
        return H/2*cos(k*(x-c*t))
