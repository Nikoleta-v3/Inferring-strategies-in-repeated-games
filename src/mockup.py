# %%
import itertools

import axelrod as axl
import numpy as np
#%%
from axelrod.action import Action

import stationary

C, D = Action.C, Action.D

PARAMETERS = {
    "epsilon": 0.00,
    "delta": 0.99,
    "seq_size": 2,
    "benefit": 1,
    "cost": 0.2
}

delta = PARAMETERS["delta"]
epsilon = PARAMETERS["epsilon"]
seq_size = PARAMETERS["seq_size"]
benefit = PARAMETERS["benefit"]
cost = PARAMETERS["cost"]
action_map = {1: C, 0: D}

# %%
def print_strategy(idx):
    if idx < 0 or idx > 31:
        raise ValueError("idx must be between 0 and 31")
    init = C if idx & 0b10000 > 0 else D
    pcc =  C if idx & 0b01000 > 0 else D
    pcd =  C if idx & 0b00100 > 0 else D
    pdc =  C if idx & 0b00010 > 0 else D
    pdd =  C if idx & 0b00001 > 0 else D
    print(f"init: {init}, pcc: {pcc}, pcd: {pcd}, pdc: {pdc}, pdd: {pdd}")
    return (init, pcc, pcd, pdc, pdd)

print_strategy(0b00000)  # ALLD-d
print_strategy(25)  # WSLS-c
print_strategy(10)  # TFT-d

# %%

def calculate_payoff_matrix(benefit, cost, delta, epsilon):
    """Calculates the payoffs $\pi(i, j)$ for i and j in Delta.

    Returns
    -------
    np.array
        A 32x32 matrix where each entry i, j is the long term payoff i receives against j.
    """

    payoff_matrix = np.zeros((32, 32))

    # For now we only consider pure memory - one strategies
    pure_memory_one_strategies = list(itertools.product([0, 1], repeat=5))

    for i, player in enumerate(pure_memory_one_strategies):
        for j, co_player in enumerate(pure_memory_one_strategies):
            ss = stationary.stationary(player, co_player, epsilon=epsilon, delta=delta)
            payoff_matrix[i, j] = ss @ np.array([benefit - cost, -cost, benefit, 0])

    return payoff_matrix

mat = calculate_payoff_matrix(benefit, cost, delta, epsilon)
print(mat)
print(mat.shape)  # => (32,32)
mat[0][0]    # => 0.0 (AllD vs AllD)
mat[31][31]  # => benefit-cost (AllC vs AllC)
mat[0][31]   # => benefit (AllD vs AllC)
mat[31][0]   # => -cost (AllC vs AllD)
mat[25][25]  # benefit-cost (WSLS-c vs WSLS-c)
mat[8] == mat[0] # GT-d behaves like AllD

# %%
def posterior_distribution(history):
    """
    Infer the co-player's strategy based on the history of the game.
    history is a list of tuples (focal player, co-player) such as
        [(C, C), (C, D), (D, C), (D, D)].
    Returns a list of posterior probabilities of the co-player's strategy.
        [p_0, p_2, ..., p_15]
    We assume that the co-player's strategy is pure memory-one and there is no implementation error.
    If an inconsistent history is given, it raises ValueError.
    """
    # strategy is described by (p_cc, p_cd, p_dc, p_dd)
    # index = 8 * p_cc + 4 * p_cd + 2 * p_dc + p_dd
    prior = [1.0/16.0] * 16
    # for two successive turns
    for turn in range(1, len(history)):
        prev = history[turn-1]  # previous moves
        curr = history[turn][1] # co-players' current move
        if prev == (C,C):
            if curr == C:
                for i in range(16):
                    if i & 0b1000 == 0:
                        prior[i] = 0
            if curr == D:
                for i in range(16):
                    if i & 0b1000 != 0:
                        prior[i] = 0
        if prev == (C,D):
            if curr == C:
                for i in range(16):
                    if i & 0b0100 == 0:
                        prior[i] = 0
            if curr == D:
                for i in range(16):
                    if i & 0b0100 != 0:
                        prior[i] = 0
        if prev == (D,C):
            if curr == C:
                for i in range(16):
                    if i & 0b0010 == 0:
                        prior[i] = 0
            if curr == D:
                for i in range(16):
                    if i & 0b0010 != 0:
                        prior[i] = 0
        if prev == (D,D):
            if curr == C:
                for i in range(16):
                    if i & 0b0001 == 0:
                        prior[i] = 0
            if curr == D:
                for i in range(16):
                    if i & 0b0001 != 0:
                        prior[i] = 0
    # normalize prior
    if sum(prior) == 0:
        raise ValueError("Inconsistent history")
    prior = np.array([p/sum(prior) for p in prior])
    return prior


# history = [(C,C)]
# posterior_distribution(history)  #=> [1/16] * 16

# history = [(C,C),(D,D)]
# posterior_distribution(history)   #=> [1/8] * 8 + [0] * 8

# history = [(C,C),(D,D),(D,C)]
# posterior_distribution(history)   # => [0, 0.25] * 4 + [0] * 8

history = [(C,C),(D,D),(D,C),(C,D),(D,C)]
posterior_distribution(history)   # => [0,0,0,0,0,1,0,0,...0]

# history = [(D,D),(D,D),(D,C)]   # for inconsistent history
# posterior_distribution(history)   # => raise Exeception
# %%
def infer_best_response_and_expected_payoffs(history, payoff_matrix):
    """Based on a given initial sequences (history) we try to infer the strategy
    of the co-player.
    
    We calculate the posterior distribution given that co-player's
    strategy is in Delta. Namely, the posterior distribution: $p(i)$, where $i$
    is the index of the strategy $[1, 16]$.

    We then calculate the long term payoffs for the player $\pi(i, j)$
    of strategy $i$ against strategy $j$.
    Here, the we consider the case where the initial moves are history[-1] because players continue the game after the initial moves.

    If the focal player takes strategy $1$, for instance, the expected long-term 
    payoff $P(1) = $\sum_i \pi(1, i) p(i)$.

    In general, $P(j) = \sum_i \pi(j, i) p(i)$. We want to find the strategy $j$
    that maximizes $P(j)$. Namely, $j = \argmax P(j)$.
    """

    posterior = posterior_distribution(history)
    last_coplayer_move = history[-1][1]
    # we consider a repeated game starting from t=(len(history)-1)
    zero_16 = np.full((16,), 0.0)
    if last_coplayer_move == C:
        posterior = np.concatenate((zero_16, posterior))
    else:
        posterior = np.concatenate((posterior, zero_16))

    expected_payoffs = np.sum(payoff_matrix * posterior, axis=1)
    # expected_payoffs[i] = \sum_j \pi(i, j) p(j)
    # expected_payoff when the focal player takes strategy i

    last_focal_player_move = history[-1][0]
    if last_focal_player_move == C:
        # we have to choose the best response from [16, 31]
        expected_payoffs[0:16] = -np.inf
    else:
        # we have to choose the best response from [0, 15]
        expected_payoffs[16:32] = -np.inf
    # print(expected_payoffs)

    bs = np.argmax(expected_payoffs)
    exp_p = np.max(expected_payoffs)

    return bs, exp_p

mat = calculate_payoff_matrix(benefit, cost, delta, epsilon)
bs,exp_p = infer_best_response_and_expected_payoffs([(C,C),(D,C),(C,D),(C,C)], mat)
# bs,exp_p = infer_best_response_and_expected_payoffs([(D,C),(C,D),(D,D),(C,C),(C,C)], mat)   # WSLS-c
bs,exp_p

# %%
def long_term_total_payoff(
    opening_payoffs, exp_p, delta
):
    """Compute the long term payoffs of the strategy against the opponent."""
    payoffs = 0
    for turn, turn_payoff in enumerate(opening_payoffs):
        payoffs += turn_payoff[0] * delta ** turn
    # print(payoffs, exp_p)
    return payoffs + exp_p * delta ** len(opening_payoffs) / (1.0-delta)

# %%

if __name__ == "__main__":

    # define game with benefit and cost
    donation = axl.game.Game(r=benefit - cost, s=-cost, t=benefit, p=0)

    initial_sequences = list(itertools.product(["C", "D"], repeat=seq_size))

    payoff_matrix = calculate_payoff_matrix(benefit, cost, delta, epsilon)
    for init_seq in initial_sequences:

        s = axl.Cycler("".join(init_seq))
        pure_memory_one_strategies = itertools.product([0, 1], repeat=5)
        total_payoff = 0

        for i in pure_memory_one_strategies:
            opponent = axl.MemoryOnePlayer(i[1:], initial=action_map[i[0]])

            # simulating game
            match = axl.Match(players=(s, opponent), turns=len(init_seq), game=donation)
            _ = match.play()
            history = match.result
            opening_payoffs = match.scores()

            # inferring co-player and best response
            bs, exp_p = infer_best_response_and_expected_payoffs(history, payoff_matrix)
            # exclude the last payoff because it is included in exp_p
            lt_payoffs = long_term_total_payoff(
                opening_payoffs[:-1], exp_p, delta
            )
            total_payoff += lt_payoffs

        print(f"{init_seq} {total_payoff}")
        f = open(f"{''.join(init_seq)}.txt", "w")
        f.write(f"{benefit}, {cost}, {delta}, {epsilon}, {total_payoff}")
        f.close()
# %%
