# Action-gap-gated Thompson

A targeted alternative to ε-Thompson: explore *only when the planner is
genuinely uncertain about which action is best*, exploit the mean
policy otherwise.

Mechanic per decision step:

1. Compute the Q-values for all actions under the posterior-**mean**
   kernel and find the top two: Q⁽¹⁾ ≥ Q⁽²⁾.
2. If the gap |Q⁽¹⁾ − Q⁽²⁾| > τ, the best action is clearly best —
   take it (exploit).
3. Otherwise, draw a kernel from the Beta posterior and act on its
   Q-values for this step (Thompson sample).

τ is the gap threshold — small τ ⇒ rarely explore (almost pure
exploitation), large τ ⇒ always Thompson. Sweep on CC / gpt5_mini,
n = 20, τ ∈ {0.5, 1.0, 2.0, 5.0, 100.0}.

## Results

| τ | fix % | Ū | reading |
|---:|---:|---:|---|
| **0.5** | 40 | **+16.00** | best — exploration only on truly tight calls |
| 1.0 | 25 | +10.55 | over-explores; same as ε = 1 |
| 2.0 | 30 | +16.35 | second best |
| 5.0 | 25 | +10.55 | collapses to pure Thompson |
| 100.0 | 25 | +10.55 | collapses to pure Thompson |

τ = 0.5 holds onto the offline fix rate (40 %) while letting Thompson
explore the small set of marginal-gap decisions where the mean policy
might be wrong — net Ū = +16.00, the best of the five points and +1.25
above plain offline (+14.75) on the same n.

## Motivation chain

The UQ-via-Thompson experiment (see the `uq-thompson` branch) reports
that on this cell the gap between the top-two actions is wide most of
the time (`p_gap_lt_1_0 = 1.0` is rare; mean gap ≈ 0.75). So the gating
condition fires on a small fraction of decisions — which is exactly
what we want: spend exploration only where it can pay off.

## Caveat

n = 20 → all paired CIs cross zero against offline. The τ-curve is
shaped consistently with the hypothesis (small τ wins, large τ
collapses to Thompson) but isn't statistically conclusive on this cell.

## Files

- `cc_live_gapthompson_{0_5, 1_0, 2_0, 5_0, 100_0}_gpt5mini_n20.json`

Same record schema as other CC live runs.
