# procedure

Based on a given initial sequences, we calculate the posterior distribution.

- The posterior distribution: $p(i)$, where $i$ is the index of the strategy $[1,32]$
- $\pi(i, j)$: the long-term payoff of strategy $i$ against strategy $j$

If the focal player takes strategy $1$, for instance, the expected long-term payoff $P(1) = $\sum_i \pi(1,i) p(i)$.
In general, $P(j) = \sum_i \pi(j,i) p(i)$.
We want to find the strategy $j$ that maximizes $P(j)$.
$j = \argmax P(j)$.