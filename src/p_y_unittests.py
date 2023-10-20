import unittest
import numpy as np
from src.p_y_sand import *
from src.p_y_clay import *

class TestUnifiedCPT(unittest.TestCase):

    def test_p_y_sand_monotonic(self):
        '''
        Test p-y curve for sand under monotonic loading using unified CPT method
        reference data: p-y curves CPT4.xlsx, monotonic, D=1.0m, qt = 0.0955, sigma_v = 12.20290, sigma_v_e = 2.20290
        '''
        phi_e = calc_phi_e(qt = 0.0955, sigma_v = 12.20290, sigma_v_e = 2.20290)
        self.assertAlmostEqual(phi_e, 34.95398, 3)
        C1 = calc_C1(phi_e)
        self.assertAlmostEqual(C1, 2.958448, 3)
        C2 = calc_C2(phi_e)
        self.assertAlmostEqual(C2, 3.41142, 3)
        C3 = calc_C3(phi_e)
        self.assertAlmostEqual(C3, 53.47685, 3)
        k = calc_k(phi_e)
        self.assertAlmostEqual(k, 21877.60, 2)
        pr = calc_pr(D = 2, gamma = 12.20290, z = 0.98, C1 = C1, C2 = C2, C3 = C3)
        self.assertAlmostEqual(pr, 20.98854, 3)
        A = calc_A(D = 2, z = 0.98, loading = 'Monotonic')
        self.assertAlmostEqual(A, 2.608, 3)
        p1 = calc_p(y = 10, A = A, pr = pr, z = 0.98, k =k)
        self.assertAlmostEqual(p1, 54.69477, 3)
    
    def test_p_y_sand_cyclic(self):
        '''
        Test p-y curve for sand under cyclic loading using unified CPT method
        reference data: p-y curves CPT4.xlsx, cyclic, D=1.0m, qt = 0.0955, sigma_v = 12.20290, sigma_v_e = 2.20290
        '''
        phi_e = calc_phi_e(qt = 0.0955, sigma_v = 12.20290, sigma_v_e = 2.20290)
        self.assertAlmostEqual(phi_e, 34.95398, 3)
        C1 = calc_C1(phi_e)
        self.assertAlmostEqual(C1, 2.958448, 3)
        C2 = calc_C2(phi_e)
        self.assertAlmostEqual(C2, 3.41142, 3)
        C3 = calc_C3(phi_e)
        self.assertAlmostEqual(C3, 53.47685, 3)
        k = calc_k(phi_e)
        self.assertAlmostEqual(k, 21877.60, 2)
        pr = calc_pr(D = 2, gamma = 12.20290, z = 0.98, C1 = C1, C2 = C2, C3 = C3)
        self.assertAlmostEqual(pr, 20.98854, 3)
        A = calc_A(D = 2, z = 0.98, loading = 'Cyclic')
        self.assertAlmostEqual(A, 0.9, 3)
        p1 = calc_p(y = 10, A = A, pr = pr, z = 0.98, k =k)
        self.assertAlmostEqual(p1, 18.88969, 3)

    def test_p_y_clay_monotonic(self):
        '''
        Test p-y curve for clay under monotonic loading using unified CPT method
        reference data: p-y curves CPT4.xlsx, monotonic, D=2.0m, z=3.0, qt = 0.0955, sigma_v = 12.20290, sigma_v_e = 2.20290
        '''
        su = calc_su(qt = 0.15720, sigma_v = 42.35137, nkt = 12)
        self.assertAlmostEqual(su, 9.57072, 3)
        su1 = calc_su1(su = 9.57072, su0 = 0.63357, z = 2.980)
        self.assertAlmostEqual(su1, 2.99904, 3)
        alpha = calc_alpha(su = 9.57072, sigma_v_e = 12.35137)
        self.assertAlmostEqual(alpha, 0.56801, 3)
        N_pd = calc_N_pd(alpha = 0.56801)
        self.assertAlmostEqual(N_pd, 10.70403, 3)
        d = calc_d(su0 = 0.63357, su1 = 2.99904, D = 2.0)
        self.assertAlmostEqual(d, 19.04530, 3)
        N_p0 = calc_N_p0(N_1 = 12, N_2 = 3.22, alpha = 0.56801, d = 19.04530, D = 2.0, N_pd = 10.70403, z = 2.980)
        self.assertAlmostEqual(N_p0, 5.25509, 3)
        N_P = calc_N_P(N_pd = 10.70403, N_p0 = 5.25509, gamma = 14.11712, z = 3.0, su = 9.57072, isotropy = 'true')
        self.assertAlmostEqual(N_P, 6.54563, 3)
        pu = calc_pu(su = 9.57072, D = 2.0, N_P = 6.54563)
        self.assertAlmostEqual(pu, 125.29273, 3)

if __name__ == "__main__":
    unittest.main()