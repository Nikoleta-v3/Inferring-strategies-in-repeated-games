# %%

PARAMETERS = {
    'epsilon': 0.001,
    'delta': 0.99
}

delta = PARAMETERS['delta']
epsilon = PARAMETERS['epsilon']

class BeyesianBestResponseStrategy:
    def __init__(self, initial_sequence):
        self.initial_sequence = initial_sequence

    def best_response_and_expected_payoffs(self, history):
        """Compute the best response based on the history."""
        # [TODO] implement me

    def posterior_distribution(self, history):
        """Compute the posterior distribution of the opponent's strategy."""
        # [TODO] implement me

def long_term_payoffs(strategy, opponent, history, opening_payoffs, delta, epsilon):
    """Compute the long term payoffs of the strategy against the opponent."""
    # [TODO] implement me


for init_seq in initial_sequences:
    s = BayesianBestResponseStrategy(init_seq)

    total_payoff = 0
    for i in range(16):
        opponent = MemoryOneStrategy(i)
        opening_payoffs, history = match(s, opponent, len(init_seq))
        bs,exp_p = s.best_response_and_expected_payoffs(history)
        lt_payoffs = long_term_payoffs(bs, opponent, history, opening_payoffs, delta, epsilon)
        total_payoff += lt_payoffs[0]

    print(f"{init_seq} {total_payoff}")

