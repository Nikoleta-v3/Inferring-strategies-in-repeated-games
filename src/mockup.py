# %%
import itertools
import bayesian
import axelrod as axl

#%%
from axelrod.action import Action

C, D = Action.C, Action.D

PARAMETERS = {
    "epsilon": 0.001,
    "delta": 0.99,
    "seq_size": 2,
}

delta = PARAMETERS["delta"]
epsilon = PARAMETERS["epsilon"]
seq_size = PARAMETERS["seq_size"]
action_map = {1: C, 0: D}

# %%

class BayesianBestResponseStrategy:
    def __init__(self, initial_sequence):
        self.initial_sequence = initial_sequence

    def best_response_and_expected_payoffs(self, history):
        """Compute the best response based on the history."""
        # [TODO] implement YM

    def posterior_distribution(self, history):
        """Compute the posterior distribution of the opponent's strategy."""
        num_possible_s = 32
        last_turn_outcomes = ["p0"] + list(itertools.product([C, D], repeat=2))
        pure_transitions = list(itertools.product([C, D], repeat=5))
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
                strategy.likelihood(turn_action, history[turn])
                for strategy in strategies_to_fit
            ]

            posteriors = bayesian.posterior(priors, likelihoods)

            priors = posteriors

        return priors


def long_term_payoffs(
    strategy, opponent, history, opening_payoffs, delta, epsilon
):
    """Compute the long term payoffs of the strategy against the opponent."""
    # [TODO] implement NG

# %%
initial_sequences = itertools.product([C, D], repeat=seq_size)
list(initial_sequences)

# %%

if __name__ == "__main__":

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
            opening_payoffs = match.final_score_per_turn()

            # inferring co-player and best response
            bs, exp_p = infer_best_response_and_expected_payoffs(history)
            bs, exp_p = s.best_response_and_expected_payoffs(history)

            lt_payoffs = long_term_payoffs(
                bs, opponent, history, opening_payoffs, delta, epsilon
            )
            total_payoff += lt_payoffs[0]

        print(f"{init_seq} {total_payoff}")

# %%
init_seq = [C,D,C]
init_seq_str = "".join([a.name for a in init_seq])
axl.Cycler(init_seq_str)


# %%
pure_memory_one_strategies = itertools.product([0, 1], repeat=5)
i = list(pure_memory_one_strategies)[10]
#for i in pure_memory_one_strategies:
opponent = axl.MemoryOnePlayer(i[1:], initial=action_map[i[0]]) # %%
opponent

# %%

def infer_best_response_and_expected_payoffs(history):

    return bs, exp_p
