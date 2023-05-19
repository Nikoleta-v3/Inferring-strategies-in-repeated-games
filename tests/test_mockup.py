# At the moment relative paths matter so we need this line.
# At some point we will create a Python module for our code and this won't
# be necessary

import sys

sys.path.append("src/")

import mockup as mcu

import numpy as np

from axelrod.action import Action

C, D = Action.C, Action.D


def test_calculate_payoff_matrix():
    benefit, cost, delta, epsilon = 2, 1, 0.999, 0

    matrix = mcu.calculate_payoff_matrix(benefit, cost, delta, epsilon)

    # Dimension
    assert matrix.shape == (32, 32)

    # ALLD
    assert np.isclose(matrix[0, 0], 0)
    assert np.isclose(matrix[0, 31], 2)

    # ALLC
    assert np.isclose(matrix[31, 0], -1)
    assert np.isclose(matrix[31, 26], 1)  # TFT (C)
    assert np.isclose(matrix[31, 10], 0.998)  # TFT (D)

    # TFT
    assert np.isclose(matrix[26, 26], 1)
    assert np.isclose(matrix[26, 25], 1)  # WSLS


def test_posterior_distribution_TFT_C():
    history = [(C, C), (D, C), (C, D), (D, C), (D, D), (D, D)]
    posterior = mcu.posterior_distribution(history)
    assert np.argmax(posterior) == 26


def test_posterior_distribution_TFT_D():
    history = [(C, D), (C, C), (D, C), (D, D), (D, D), (D, D)]
    posterior = mcu.posterior_distribution(history)
    assert np.argmax(posterior) == 10


def test_posterior_distribution_WSLS_C():
    history = [(C, C), (D, C), (C, D), (D, D), (D, C)]
    posterior = mcu.posterior_distribution(history)
    assert np.argmax(posterior) == 25


def test_posterior_distribution_WSLS_D():
    history = [(C, D), (D, D), (C, C), (C, C), (D, C), (D, D)]
    posterior = mcu.posterior_distribution(history)
    assert np.argmax(posterior) == 9
