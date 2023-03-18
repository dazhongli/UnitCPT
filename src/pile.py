import numpy as np
import pandas as pd
from numpy import pi
from numpy import degrees, radians, tan
from src.soil import Stratum
import src.utilities as ult
from numpy import vectorize
from copy import copy


def fs(sigma_v, su, alpha, beta, drainage, reduction_ud=1.0, reduction_dr=1.0):
    return np.where(drainage == 'UD', alpha*su*reduction_ud, beta*sigma_v*reduction_dr)


class Pile:
    def __init__(self):
        self.E = 205E6  # in kPa


class PipePile(Pile):
    def __init__(self, dia, thickness, length, corrosion=0):
        '''
        Initialise the pile with diameter and thickness
        All the thickness input should be in 'm'
        '''
        super(PipePile, self).__init__()
        self.dia_out = dia
        self.t = thickness
        self.corrosion = corrosion
        self.length = length
        self.refresh()

        ###
    def refresh(self):
        self.corroded_dia = self.dia_out - 2*self.corrosion
        self.dia_inner = self.dia_out - 2*self.t
        self.perimeter_inner = pi*self.dia_inner
        self.perimeter_outer = pi*self.dia_out
        self.gross_area = 1/4*pi*(self.corroded_dia)**2
        self.inner_area = 1/4*pi*(self.dia_inner)**2
        self.annulus_area = 1/4*pi*(self.corroded_dia**2 - self.dia_inner**2)
        self.disp_ratio = self.annulus_area/self.gross_area
        self.weight_dry = self.annulus_area * self.length * 78.5
        self.weight_submerged = self.weight_dry*68.5/78.5

    def can_weight(self, cover_thickness=0.09, submerged=True):
        '''
        return the weight of the can, i.e., the upper end is closed with steel plate of thickness 'cover_thickness
        '''
        if submerged:
            return self.weight_submerged + cover_thickness * self.inner_area * 68.5
        else:
            return self.weight_dry + cover_thickness * self.inner_area * 78.5

    def Qc(self, soil: Stratum, toe_level, a, b, qb, reduction_ud=1.0, reduction_dr=1.0, mode='unplugged'):
        '''
        This function calculate the pile capacity using the alpha and beta method
        a = alpha
        b = beta
        qb = base resistance
        '''

        # we reserve an orignal copy of the data
        df_original = copy(soil.df_soil)
        df = soil.df_soil
        df = ult.interp_row(df, toe_level, 'bottom_level', method='ffill')
        df = ult.interp_row(df, toe_level, 'bottom_level', method='bfill')
        soil.df_soil = df
        soil.refresh()
        df = copy(soil.df_soil)
        a = np.ones(df.shape[0])*a
        b = np.ones(df.shape[0])*b
        df['alpha'] = a
        df['beta'] = b

        df['tau_s'] = df.apply(lambda row: fs(
            row.sigma_v_e, row.su, row.alpha, row.beta, row.drainage, reduction_ud=reduction_ud, reduction_dr=reduction_dr), axis=1)
        df['Fs_i'] = df.tau_s*self.perimeter_inner
        df['Fs_o'] = df.tau_s*self.perimeter_outer

        df_pile = df[df.bottom_level <= toe_level]

        Q_outside = np.trapz(df_pile.Fs_o, df_pile.bottom_level)
        Q_inside = np.trapz(df_pile.Fs_i, df_pile.bottom_level)
        Qb_wall = self.annulus_area*qb

        if mode == 'unplugged':
            return dict(Q_outside=Q_outside,
                        Q_inside=Q_inside,
                        Qb_wall=Qb_wall,
                        Q_total=Q_outside+Q_inside+Qb_wall,
                        Detail_calc=df_pile)
        soil.df_soil = df_original


class PileCapacityCPT():
    def __init__(self, cpt_data):
        self.cpt_data = cpt_data
        self.pile = None

    def Qb(self, qc, qc_average, pile, Dr, method='UWA'):
        pa = 101
        Ar = pile.disp_ratio
        qc = 1000*qc
        qc_average = 1000*qc_average
        D = pile.dia_out
        Di = pile.dia_inner
        Dcpt = 0.036  # for a standard cone with base area of 10cm2
        if method == 'UWA':
            q = qc_average * (0.15 + 0.45 * Ar)
        elif method == 'ICP':
            q = max((0.5-0.25*np.log10(D/Dcpt)), 0.15)*qc_average
            if Di > 2*(Dr - 0.3) or Di/Dcpt > 0.083*qc/pa:
                return qc * pile.annulus_area
            else:
                return pile.gross_area * q
        else:
            print('Not Implemented!')

    def su(self, name, z):
        if name == 'MD':
            if z <= 2.0:
                return 3.0
            else:
                return 3+(z-2.0)*0.5
        elif name == 'UA':
            return 1.33*z
        else:  # we are dealing with sand, and will be using the CPT data to get the friction
            return np.nan

    def fz(self, name, z, drainage, qc, sigma_v, delta, pile, method='UWA', b_compression=True):
        '''
        return the shaft friction of the tubular piles following the API code
        qc - cone resistance
        sigma_v - effective vertical stress in kPa
        pile - the pile class hold the basic information of pile
        delta - the delta_cv, the constant volume friction between the pile and soil
        '''
        qc = qc*1000  # qc in MPa
        if drainage == 'UD':
            return self.su(name, z)
        delta = radians(delta)
        if method == 'UWA':
            a = 0
            b = 0.3
            c = 0.5
            d = 1
            e = 0
            v = 2.0
            u = 0.03 if b_compression else 0.022
        elif method == 'ICP':
            a = 0.1
            b = 0.2
            c = 0.4
            d = 1.0
            e = 0
            v = 4*(pile.disp_ratio)**0.5
            u = 0.023 if b_compression else 0.016
        else:
            print('Not Implemented')
        Ar = pile.disp_ratio
        D = pile.dia_out
        L = pile.embedment
        depth_ratio = (L-z)/D
        c1 = u*qc*(sigma_v/100)**a
        c2 = max(depth_ratio, v)**(-c)
        c3 = min(depth_ratio/v, 1)**e
        fs = c1 * Ar**b * c2 * tan(delta)**d * c3
        return fs

    def pile_compression_capacity(self, method='UWA'):
        if self.pile is None or self.cpt_data is None:
            raise Exception(
                'Either Pile information or CPT data not ')


class PileDriving:
    '''
    The class handle's Alm's method which consider the friction fatigue of shaft resistance
    during pile driving
    '''

    def __init__(self, cpt_data=None, pile=None, soil: Stratum = None):
        self.cpt_data = cpt_data
        self.pile = pile
        self.soil = soil

    def shape_factor(self, qc: float, sigma_v: float) -> float:
        return sqrt(qc*1000/sigma_v)/80

    def kh(self, qc, sigma_v):
        '''
        Return the horizontal coefficient
        qc - cone resistant at the tip in MPa
        sigma_v, effective vertical stress in kPa
        '''
        return 0.0132*qc*(sigma_v/100)**0.13/sigma_v * 1000

    def init_friction(self, cone_friction, kh, sigma_v, phi, drainage='Dr'):
        '''
        Return the initial shaft friction based on ALM's method
        this will included both internal and externall
        kh : horizontal soil pressure coefficient
        sigma_v: effective vertical stress
        cone-friction : Cone friction measured from the CPT tests
        drainage: 'Dr' or 'UD' for the drained or undrained case, respecrtivley:
        '''
        if drainage == 'UD':
            return cone_friction
        else:  # Drained Condition
            try:
                return kh * sigma_v*np.tan(np.radians(phi))
            except:
                print(phi)

    def residual_friction(self, qc, sigma_v: float, init_friction: float, drainage: str) -> float:
        if drainage == 'Dr':
            return 0.2 * init_friction
        else:  # undrained material
            try:
                return 0.004*qc*1000*(1-0.0025*qc*1000/sigma_v)
            except:
                return np.nan

    def Qb(self, qc, sigma_v, drainage):
        '''
        Returns the cone resistance
        qc - cone resistance at the tip in MPa
        sigma_v - effective vertical stress in kPa
        '''
        assert (self.pile is not None)
        Ab = self.pile.annulus_area
        if drainage == 'UD':
            return 0.6*qc*1000 * Ab
        else:
            return 0.15*qc*(qc/sigma_v*1000)**0.2*1000 * Ab

    def shaft_friction(self, init_friction, residual_friction,
                       shape_factor, depth, pile_toe_loc):
        '''
        Return the shaft friction during driving based on Alm's method.
        It proportion residual and initial friction based on the location relative to the  *np.exp(t*(d-p))
        toe level. 
        '''
        t = shape_factor
        d = depth
        p = pile_toe_loc
        fi = init_friction
        fres = residual_friction
        return fres + (fi - fres) * np.exp(t*(d-p))

    def soil_static_resistance(self, level):
        '''
        Return the pile capacity, the function will check if the cpt, soil and pile information 
        has already been passed to the ALM 

        Args:
            level: the level of intests

        '''
        df = self.cpt_data
        df = self.soil.add_soil_names_df(df, 'STCN_DPTH')
        df['sigma_v'] = df.STCN_DPTH.apply(self.soil.effective_stress)
        df['shape_factor'] = self.shape_factor(df.STCN_RES,
                                               df.sigma_v)
        df['kh'] = self.kh(df.STCN_RES, df.sigma_v)
        df['phi'] = 32  # degrees
        df['fsi'] = df.apply(lambda col: self.init_friction(col.STCN_FRES,
                                                            col.kh,
                                                            col.sigma_v,
                                                            col.phi,
                                                            col.drainage), axis=1)
        df['fres'] = df.apply(lambda col: self.residual_friction(col.STCN_RES,
                                                                 col.sigma_v,
                                                                 col.fsi,
                                                                 col.drainage), axis=1)
        df['fs'] = df.apply(lambda col: self.shaft_friction(col.fsi,
                                                            col.fres,
                                                            col.shape_factor,
                                                            col.STCN_DPTH,
                                                            level), axis=1)
        df['dh'] = (df.STCN_DPTH - df.STCN_DPTH.shift(1)).fillna(0)
        df['dfs'] = df.fs * df.dh
        df['Fs'] = df.dfs.cumsum() * \
            (self.pile.dia_out) * np.pi  # Only External will be used
        df['Qb'] = df.apply(lambda col: self.Qb(col['STCN_RES'],
                                                col['sigma_v'],
                                                col['drainage'],
                                                ), axis=1)
        df['Qu'] = df.Fs + df.Qb
        df['Qu'] = df.apply(
            lambda col: np.nan if col['STCN_DPTH'] > level else col['Qu'], axis=1)
        df = df[df.STCN_DPTH < level]
        return df
