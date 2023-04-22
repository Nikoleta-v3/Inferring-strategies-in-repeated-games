"""A script that estimates the best responses for a given pure memory-one
strategy.
"""

import csv
import itertools
import sys
from importlib.machinery import SourceFileLoader

import sympy as sym
import tqdm

stationary = SourceFileLoader("stationary", "src/stationary.py").load_module()

from stationary import *

if __name__ == "__main__":
    player_idx = int(sys.argv[1])

    p0, pcc, pcd, pdc, pdd = sym.symbols(
        "p_{0}, p_{CC}, p_{CD}, p_{DC}, p_{DD}"
    )

    delta = sym.symbols("delta")

    epsilon = sym.symbols("epsilon")

    b, c = 1, sym.symbols("c")

    pure_strategies = list(itertools.product([0, 1], repeat=5))

    labels = [f"d{i}" for i, _ in enumerate(pure_strategies)]

    labels[0] = "ALLD (D)"
    labels[16] = "ALLD (C)"

    labels[15] = "ALLC (D)"
    labels[31] = "ALLC (C)"

    labels[8] = "GT (D)"
    labels[24] = "GT (C)"

    labels[9] = "WSLS (D)"
    labels[25] = "WSLS (C)"

    labels[10] = "TFT (D)"
    labels[26] = "TFT (C)"
    player = pure_strategies[player_idx]

    payoffs = []

    for coplayer in tqdm.tqdm(pure_strategies):
        ss = stationary(coplayer, player, epsilon, delta)

        payoffs.append(sum(ss @ sym.Matrix([b - c, -c, b, 0])).simplify())

    with open(f"outputs/{labels[player_idx]}_payoffs.txt", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(zip(labels, payoffs))
