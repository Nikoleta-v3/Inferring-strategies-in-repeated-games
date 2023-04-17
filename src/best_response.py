import csv
import itertools
import sys

import sympy as sym
import tqdm


def matrix(player, opponent):
    """Transition matrix for prisoner's dilemma when player and opponent use
    memory-one strategies.

    Parameters
    ----------
    player : list
        Player's p strategy where p = (p0, pcc, pcd, pdc, pdd)
    opponent : list
        Opponent's q strategy where q = (q0, qcc, qcd, qdc, qdd)

    Returns
    -------
    sym.Matrix
        The transition matrix
    """
    return sym.Matrix(
        [
            [
                player[1] * opponent[1],
                player[1] * (1 - opponent[1]),
                (1 - player[1]) * opponent[1],
                (1 - player[1]) * (1 - opponent[1]),
            ],
            [
                player[2] * opponent[3],
                player[2] * (1 - opponent[3]),
                (1 - player[2]) * opponent[3],
                (1 - player[2]) * (1 - opponent[3]),
            ],
            [
                player[3] * opponent[2],
                player[3] * (1 - opponent[2]),
                (1 - player[3]) * opponent[2],
                (1 - player[3]) * (1 - opponent[2]),
            ],
            [
                player[4] * opponent[4],
                player[4] * (1 - opponent[4]),
                (1 - player[4]) * opponent[4],
                (1 - player[4]) * (1 - opponent[4]),
            ],
        ]
    )


def stationary(player, coplayer, epsilon, delta):
    """Stationary distribution of a finite repeated game (with discount factor)
    between two memory-one strategies.

    Parameters
    ----------
    player : list
        Player's p strategy where p = (p0, pcc, pcd, pdc, pdd)
    opponent : list
        Opponent's q strategy where q = (q0, qcc, qcd, qdc, qdd)
    epsilon : float
        Noise
    delta : float
        Discount factor

    Returns
    -------
    sym.Matrix
        Stationary vector
    """

    eplayer = [(i * (1 - epsilon) + (1 - i) * epsilon) for i in player]

    ecoplayer = [(i * (1 - epsilon) + (1 - i) * epsilon) for i in coplayer]

    M = matrix(eplayer, ecoplayer)

    v0 = sym.Matrix(
        [
            eplayer[0] * ecoplayer[0],
            eplayer[0] * (1 - ecoplayer[0]),
            (1 - eplayer[0]) * ecoplayer[0],
            (1 - eplayer[0]) * (1 - ecoplayer[0]),
        ]
    )

    inv = (sym.eye(4) - delta * M).inv()

    return ((1 - delta) * v0.T) @ inv


def stationary_no_discount(player, coplayer, epsilon):
    """Stationary distribution of a infinitely repeated game between two
    memory-one strategies.

    Parameters
    ----------
    player : list
        Player's p strategy where p = (p0, pcc, pcd, pdc, pdd)
    opponent : list
        Opponent's q strategy where q = (q0, qcc, qcd, qdc, qdd)
    epsilon : float
        Noise

    Returns
    -------
    sym.Matrix
        Stationary vector
    """
    eplayer = [(i * (1 - epsilon) + (1 - i) * epsilon) for i in player]

    ecoplayer = [(i * (1 - epsilon) + (1 - i) * epsilon) for i in coplayer]

    M = matrix(eplayer, ecoplayer)

    size = M.shape[0]
    pi = sym.symbols(f"b_1:{size + 1}")
    ss = sym.solve(
        [sum(pi) - 1]
        + [a - b for a, b in zip(M.transpose() * sym.Matrix(pi), pi)],
        pi,
    )

    v_vector = sym.Matrix(
        [
            [ss[p] for p in pi],
        ]
    )

    return v_vector

if __name__ == "__main__":

    player_idx = int(sys.argv[1])

    # Kim, Choi and Baek Paper
    p0, pcc, pcd, pdc, pdd = sym.symbols("p_{0}, p_{CC}, p_{CD}, p_{DC}, p_{DD}")

    delta = sym.symbols("delta")

    epsilon = sym.symbols("epsilon")

    b, c = 1, sym.symbols("c")

    pure_strategies = list(itertools.product([0, 1], repeat=4))

    labels = [f"d{i}" for i, _ in enumerate(pure_strategies)]
    labels[0] = "ALLD"; labels[8] = "GT"; labels[9] = "WSLS"
    labels[10] = "TFT"; labels[15] = "ALLC"

    player = list(pure_strategies[player_idx])

    payoffs = []

    for coplayer in tqdm.tqdm(pure_strategies):
        
        ss = stationary_no_discount([0] + list(coplayer), [0] + player, epsilon)
        
        payoffs.append(sum(ss @ sym.Matrix([b - c, -c, b, 0])).simplify())

    with open(f"{labels[player_idx]}_payoffs.txt", 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(labels, payoffs))