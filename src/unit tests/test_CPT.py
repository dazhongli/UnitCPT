import unittest
import ../cpt
import ../pile
import numpy as np


def su_func(name, z):
    if name == 'MD':
        if z <= 2.0:
            return 3.0
        else:
            return 3+(z-2.0)*0.5
    elif name == 'UA':
        return 1.33*z
    else:  # we are dealing with sand, and will be using the CPT data to get the friction
        return np.nan


class TestCPT(unittest.TestCase):

    def test_shaft_friction_UWA(self):
        '''Test Shaft using UWA method'''
        test_pile = pile.PipePile(1.83, 0.05, 10)
        test_pile.embedment = 64
        test_cpt = cpt.CPT()
        fs = test_cpt._shaft_friction(z=30, qt=10, sigma_v_e=100, delta=28.5, Ic=1.0,
                                      pile=test_pile, method=cpt.CPTMethod.UWA_05, b_compression=True)
        self.assertAlmostEqual(fs, 19.29017, 3)

    def test_shaft_friction_ICP(self):
        '''Test the Shaft using ICP method'''
        test_pile = pile.PipePile(1.83, 0.05, 10)
        test_pile.embedment = 64
        test_cpt = cpt.CPT()
        fs = test_cpt._shaft_friction(
            z=30, qt=10, sigma_v_e=200, delta=28.5, Ic=1.0, pile=test_pile, method=cpt.CPTMethod.ICP_05, b_compression=True)
        self.assertAlmostEqual(fs, 26.564, 3)
        # refer to Eq.C.5

    def test_shaft_friction_Fugro(self):
        '''Test the Shaft using ICP method'''
        test_pile = pile.PipePile(1.83, 0.05, 10)
        test_pile.embedment = 64
        test_cpt = cpt.CPT()
        fs = test_cpt._shaft_friction(
            z=30, qt=10, sigma_v_e=200, delta=28.5, Ic=1.0, pile=test_pile, method=cpt.CPTMethod.FUGRO_05, b_compression=True)
        self.assertAlmostEqual(fs, 11.704, 3)
        # refer to Eq.C.5

    def test_toe_resitance_UWA(self):
        '''Test the toe resistance using the UWA method'''
        test_pile = pile.PipePile(1.83, 0.05, 10)
        test_pile.embedment = 64
        test_cpt = cpt.CPT()
        Qb = test_cpt._base_Qb(25, 20, test_pile, 0.6,
                               method=cpt.CPTMethod.UWA_05)
        self.assertAlmostEqual(Qb, 10407, 0)

    def test_toe_resitanct_ICP(self):
        '''Test the toe resistance using the ICP method'''
        test_pile = pile.PipePile(1.83, 0.05, 10)
        test_pile.embedment = 64
        test_cpt = cpt.CPT()
        Qb = test_cpt._base_Qb(25, 20, test_pile, 0.6,
                               method=cpt.CPTMethod.ICP_05)
        self.assertAlmostEqual(Qb, 6990, 0)

    def test_toe_resitanct_ICP(self):
        '''Test the toe resistance using the ICP method'''
        test_pile = pile.PipePile(1.83, 0.05, 10)
        test_pile.embedment = 64
        test_cpt = cpt.CPT()
        Qb = test_cpt._base_Qb(qc=25, qc_average=20, pile=test_pile, Dr=0.6,
                               method=cpt.CPTMethod.FUGRO_05)
        self.assertAlmostEqual(Qb, 18053.563, 0)

    def test_shaft_friction_UWA_clay(self):
        '''Test Shaft using UWA method'''
        test_pile = pile.PipePile(1.83, 0.05, 10)
        test_pile.embedment = 64
        test_cpt = cpt.CPT()
        fs = test_cpt._shaft_friction(z=5, qt=2, sigma_v_e=100, delta=28.5, Ic=3.0,
                                      pile=test_pile, method=cpt.CPTMethod.UWA_05, b_compression=True, consider_clay=True)
        self.assertAlmostEqual(fs, 38.208, 2)


if __name__ == "__main__":
    unittest.main()
