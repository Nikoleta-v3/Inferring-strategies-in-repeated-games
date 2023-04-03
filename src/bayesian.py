def init_prior_uniform(n):
    """
    Returns 1 / n where n is the number of possibles strategies.

    This is known as the prior.

    If we are fitting for n = 3, then the initial prior that a player
    is playing any of the 3 strategies is 1 / 3.
    """
    return 1 / n


def normalising_constant(priors, likelihoods):
    return sum([p * l for p, l in zip(priors, likelihoods)])


def posterior(self_prior,
              self_likelihood,
              priors,
              likelihoods):

    return (self_prior * self_likelihood) / normalising_constant(
        priors, likelihoods
    )


class MemoryOne:
    def __init__(self, states_action_dict, error=0):
        self.error = error
        self.states_action_dict = states_action_dict

    def likelihood(self, action, history):

        if (
            action
            == self.states_action_dict[history]
        ):
            return 1 * (1 - self.error)
        else:
            return self.error
