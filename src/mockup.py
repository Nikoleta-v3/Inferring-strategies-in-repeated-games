# %%
import itertools

import axelrod as axl
import numpy as np
#%%
from axelrod.action import Action

import bayesian
import stationary

C, D = Action.C, Action.D

PARAMETERS = {
    "epsilon": 0.001,
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
            payoff_matrix[i, j] = sum(ss @ np.array([benefit - cost, -cost, benefit, 0]))

    return payoff_matrix

# %%
def infer_best_response_and_expected_payoffs(history, benefit, cost, delta, epsilon):
    """Based on a given initial sequences (history) we try to infer the strategy
    of the co-player.
    
    We calculate the posterior distribution given that co-player's
    strategy is in Delta. Namely, the posterior distribution: $p(i)$, where $i$
    is the index of the strategy $[1, 32]$.

    We then calculate the long term payoffs for the player $\pi(i, j)$
    of strategy $i$ against strategy $j$

    If the focal player takes strategy $1$, for instance, the expected long-term 
    payoff $P(1) = $\sum_i \pi(1, i) p(i)$.

    In general, $P(j) = \sum_i \pi(j, i) p(i)$. We want to find the strategy $j$
    that maximizes $P(j)$. Namely, $j = \argmax P(j)$.
    """

    posterior = posterior_distribution(history)
    # For testing purpose
    print(np.argmax(posterior))
    payoff_matrix = calculate_payoff_matrix(benefit, cost, delta, epsilon)

    expected_payoffs = np.sum(payoff_matrix * posterior, axis=1)

    bs = np.argmax(expected_payoffs)
    exp_p = np.max(expected_payoffs)

    return bs, exp_p
# %%
def posterior_distribution(history):
    """Compute the posterior distribution of the opponent's strategy."""
    num_possible_s = 32
    last_turn_outcomes = ["p0"] + list(itertools.product([1, 0], repeat=2))
    pure_transitions = list(itertools.product([0, 1], repeat=5))
    pure_strategies = {
        f"M{i}": {k: v for k, v in zip(last_turn_outcomes, transitions)}
        for i, transitions in enumerate(pure_transitions)
    }
    strategies_to_fit = [
        bayesian.MemoryOne(error=epsilon, states_action_dict=value)
        for value in pure_strategies.values()
    ]

    priors = [bayesian.init_prior_uniform(num_possible_s)] * num_possible_s

    coplayers_actions = [h[1] for h in history]

    # Opening Move
    opening_move = coplayers_actions[0]

    likelihoods = [
        strategy.likelihood(opening_move, "p0")
        for strategy in strategies_to_fit
    ]
    posteriors = bayesian.posterior(priors, likelihoods)
    priors = posteriors

    # The rest
    for turn, turn_action in enumerate(coplayers_actions[1:]):
        likelihoods = [
            strategy.likelihood(turn_action, history[turn][::-1])
            for strategy in strategies_to_fit
        ]

        posteriors = bayesian.posterior(priors, likelihoods)

        priors = posteriors

    return priors
# %%

def long_term_payoffs(
    opening_payoffs, exp_p, delta
):
    """Compute the long term payoffs of the strategy against the opponent."""
    payoffs = 0
    for turn, turn_payoff in enumerate(opening_payoffs):
        payoffs += turn_payoff[0] * delta ** turn
    return payoffs + exp_p * delta ** len(opening_payoffs)

# %%
initial_sequences = itertools.product([C, D], repeat=seq_size)
list(initial_sequences)

# %%

if __name__ == "__main__":

    # define game with benefit and cost

    initial_sequences = itertools.product([C, D], repeat=seq_size)

    for init_seq in initial_sequences:
        # make string from list
        init_seq_str = "".join([a.name for a in init_seq])
        s = axl.Cycler(init_seq_str)

        pure_memory_one_strategies = itertools.product([0, 1], repeat=5)
        total_payoff = 0
        #s = BayesianBestResponseStrategy(init_seq)
        for i in pure_memory_one_strategies:
            opponent = axl.MemoryOnePlayer(i[1:], initial=action_map[i[0]])

            # simulating game
            match = axl.Match(players=(s, opponent), turns=len(init_seq))
            _ = match.play()
            history = match.result
            opening_payoffs = match.scores()

            # inferring co-player and best response
            bs, exp_p = infer_best_response_and_expected_payoffs(history)
            bs, exp_p = s.best_response_and_expected_payoffs(history)

            lt_payoffs = long_term_payoffs(
                bs, opponent, history, opening_payoffs, delta, epsilon
            )
            total_payoff += lt_payoffs[0]

        print(f"{init_seq} {total_payoff}")