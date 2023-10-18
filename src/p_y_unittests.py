import unittest
import numpy as np
import p_y_sand

class TestUnifiedCPT(unittest.TestCase):

    def test_p_y_sand_monotonic(self):
        '''
        Test p-y curve for sand under monotonic loading using unified CPT method
        reference data: p-y curves CPT4.xlsx, monotonic, D=1.0m, qt = 0.0955, sigma_v = 12.20290, sigma_v_e = 2.20290
        '''
        phi_e = p_y_sand.calc_phi_e(qt = 0.0955, sigma_v = 12.20290, sigma_v_e = 2.20290)
        self.assertAlmostEqual(phi_e, 34.95398, 3)
        C1 = p_y_sand.calc_C1(phi_e)
        self.assertAlmostEqual(C1, 2.958448, 3)
        C2 = p_y_sand.calc_C2(phi_e)
        self.assertAlmostEqual(C2, 3.41142, 3)
        C3 = p_y_sand.calc_C3(phi_e)
        self.assertAlmostEqual(C3, 53.47685, 3)
        k = p_y_sand.calc_k(phi_e)
        self.assertAlmostEqual(k, 21877.60, 2)
        pr = p_y_sand.calc_pr(D = 2, gamma = 12.20290, z = 0.98, C1 = C1, C2 = C2, C3 = C3)
        self.assertAlmostEqual(pr, 20.98854, 3)
        A = p_y_sand.calc_A(D = 2, z = 0.98, loading = 'Monotonic')
        self.assertAlmostEqual(A, 2.608, 3)
        p1 = p_y_sand.calc_p(y = 10, A = A, pr = pr, z = 0.98, k =k)
        self.assertAlmostEqual(p1, 54.69477, 3)
    
    def test_p_y_sand_monotonic(self):
        '''
        Test p-y curve for sand under cyclic loading using unified CPT method
        reference data: p-y curves CPT4.xlsx, cyclic, D=1.0m, qt = 0.0955, sigma_v = 12.20290, sigma_v_e = 2.20290
        '''
        phi_e = p_y_sand.calc_phi_e(qt = 0.0955, sigma_v = 12.20290, sigma_v_e = 2.20290)
        self.assertAlmostEqual(phi_e, 34.95398, 3)
        C1 = p_y_sand.calc_C1(phi_e)
        self.assertAlmostEqual(C1, 2.958448, 3)
        C2 = p_y_sand.calc_C2(phi_e)
        self.assertAlmostEqual(C2, 3.41142, 3)
        C3 = p_y_sand.calc_C3(phi_e)
        self.assertAlmostEqual(C3, 53.47685, 3)
        k = p_y_sand.calc_k(phi_e)
        self.assertAlmostEqual(k, 21877.60, 2)
        pr = p_y_sand.calc_pr(D = 2, gamma = 12.20290, z = 0.98, C1 = C1, C2 = C2, C3 = C3)
        self.assertAlmostEqual(pr, 20.98854, 3)
        A = p_y_sand.calc_A(D = 2, z = 0.98, loading = 'Cyclic')
        self.assertAlmostEqual(A, 0.9, 3)
        p1 = p_y_sand.calc_p(y = 10, A = A, pr = pr, z = 0.98, k =k)
        self.assertAlmostEqual(p1, 18.88969, 3)

if __name__ == "__main__":
    unittest.main()