# At the moment relative paths matter so we need this line.
# At some point we will create a Python module for our code and this won't
# be necessary

import sys,os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "../src")

sys.path.append(src_dir)

import mockup as mcu
import numpy as np

from axelrod.action import Action

C, D = Action.C, Action.D

import unittest

class TestMockup(unittest.TestCase):

    def test_calculate_payoff_matrix(self):
        benefit, cost, delta, epsilon = 2, 1, 0.999, 0

        matrix = mcu.calculate_payoff_matrix(benefit, cost, delta, epsilon)

        # Dimension
        self.assertEqual(matrix.shape, (32, 32))

        # ALLD
        self.assertAlmostEqual(matrix[0, 0], 0)
        self.assertAlmostEqual(matrix[0, 31], 2)

        # ALLC
        self.assertAlmostEqual(matrix[31, 0], -1)
        self.assertAlmostEqual(matrix[31, 26], 1)  # TFT (C)
        self.assertAlmostEqual(matrix[31, 10], 0.998)  # TFT (D)

        # TFT
        self.assertAlmostEqual(matrix[26, 26], 1)
        self.assertAlmostEqual(matrix[26, 25], 1)  # WSLS


    def test_posterior_distribution(self):
        # not enough history => uniform distribution
        history = [(C,C)]
        expected = np.array([1/16] * 16)
        np.testing.assert_allclose(mcu.posterior_distribution(history), expected)

        # when p_cc = 0
        history = [(C,C),(D,D)]
        expected = np.array([1/8] * 8 + [0] * 8)
        np.testing.assert_allclose(mcu.posterior_distribution(history), expected)

        # when p_cc = 0, p_dd = 1
        history = [(C,C),(D,D),(D,C)]
        expected = np.array([0, 0.25] * 4 + [0] * 8)
        np.testing.assert_allclose(mcu.posterior_distribution(history), expected)

        # when p_cc = 0, p_dd = 1, p_dc = 0, p_cd = 1  (0101)
        history = [(C,C),(D,D),(D,C),(C,D),(D,C)]
        expected = np.array([0,0,0,0,0,1,0,0] + [0] * 8)
        np.testing.assert_allclose(mcu.posterior_distribution(history), expected)

        history = [(D,D),(D,D),(D,C)]
        with self.assertRaises(Exception):
            mcu.posterior_distribution(history)


    def test_posterior_distribution_TFT(self):
        history = [(C, C), (D, C), (D, D), (C, D), (C, C)]
        posterior = mcu.posterior_distribution(history)
        self.assertEqual(np.argmax(posterior), 0b1100)


    def test_posterior_distribution_WSLS(self):
        history = [(C, C), (D, C), (C, D), (D, D), (C, C)]
        posterior = mcu.posterior_distribution(history)
        self.assertEqual(np.argmax(posterior), 0b1001)


    def test_idx_to_strategy(self):
        self.assertEqual(mcu.idx_to_strategy(0b00000), (D,D,D,D,D) )
        self.assertEqual(mcu.idx_to_strategy(0b11001), (C,C,D,D,C) ) # WSLS-c=25
        self.assertEqual(mcu.idx_to_strategy(0b01100), (D,C,C,D,D) ) # TFT-d=10


    def test_calculate_payoff_matrix(self):
        benefit,cost,delta,epsilon = 1.0,0.2,0.99,0
        mat = mcu.calculate_payoff_matrix(benefit, cost, delta, epsilon)
        self.assertEqual(mat.shape, (32,32))
        self.assertAlmostEqual(mat[0][0], 0.0) # AllD vs AllD
        self.assertAlmostEqual(mat[31][31], benefit-cost) # AllC vs AllC
        self.assertAlmostEqual(mat[0][31], benefit) # AllD vs AllC
        self.assertAlmostEqual(mat[31][0], -cost) # AllC vs AllD
        self.assertAlmostEqual(mat[25][25], benefit-cost) # WSLS-c vs WSLS-c
        self.assertTrue(np.all(mat[8] == mat[0])) # GT-d behaves like AllD


if __name__ == '__main__':
    unittest.main()