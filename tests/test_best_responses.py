import sympy as sym
from sympy.parsing.sympy_parser import parse_expr

import numpy as np

from importlib.machinery import SourceFileLoader

br = SourceFileLoader("br", "src/best_responses.py").load_module()

files = [f"d{i}" for i in range(32)]

files[0] = "ALLD (D)"
files[16] = "ALLD (C)"
files[15] = "ALLC (D)"
files[31] = "ALLC (C)"
files[8] = "GT (D)"
files[24] = "GT (C)"
files[9] = "WSLS (D)"
files[25] = "WSLS (C)"
files[10] = "TFT (D)"
files[26] = "TFT (C)"


def test_best_response():
    """A function that tests the `best_response`. For the payoffs of the
    pure memory-one strategies against a given memory-one strategy we have explicit
    expressions (folder `outputs`).

    This function evaluate the payoffs for every possible player for different values
    of c and delta.

    For the same player, c and delta, the list of best responses is returned
    with the function `best_response`.

    The tests checks that the set of strategies with the highest payoffs against
    the player is the same set as the one returned by the function.
    """
    c, delta, epsilon = sym.symbols("c, delta, epsilon")

    cvals = dvals = np.linspace(0 + 10 ** (-2), 1 - 10 ** (-2), 5).round(3)

    for i, filename in enumerate(files):
        print(i)
        # get expressions from files:
        with open(f"outputs/{filename}_payoffs.txt") as f:
            file = f.readlines()

        payoffs = [parse_expr(f.split("\t")[1].replace("\n", "")) for f in file]
        eto0 = [p.subs({epsilon: 0}).factor() for p in payoffs]

        # for a given value of c and delta check that the expression and the
        # functions give the same list of best responses

        for c_val in cvals:
            for d_val in dvals:
                numerical_eval_payoffs = [
                    round(float(p.subs({c: c_val, delta: d_val})), 10) for p in eto0
                ]

                idx_br = np.argwhere(
                    numerical_eval_payoffs == np.amax(numerical_eval_payoffs)
                )

                assert set(idx_br.flatten()) == set(br.best_responses(
                    i, c_val, d_val
                ))
