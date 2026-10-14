# Specification Gaming and Constrained-RL Mitigation in a Cleaning Gridworld

**Research question:** When an RL agent is trained on a proxy reward that
diverges from the true objective, does it learn to exploit that gap — and
can a constrained-RL formulation (Lagrangian-penalized) close it?

## Setting

`CleaningGridWorld` is a gridworld where a cheap sensor-based proxy reward
("does this cell *look* clean?") diverges from the true objective ("was
the dirt actually removed?"). The agent can either genuinely vacuum dirt
(carry it to a disposal bin — true removal) or cover it with a rug (hide
it from the sensor without removing it). Both actions raise the proxy
reward; only one raises the true reward.

## Results so far

- **Baseline** (proxy-only DQN): converges to near-maximal proxy reward
  while true reward stays at **zero** — the agent independently discovers
  two gaming mechanisms (covering dirt, and picking dirt up without ever
  delivering it) without being told either exists.
- **Constrained** (fixed-λ Lagrangian penalty on a cost signal that fires
  on covering or end-of-episode abandonment, plus a protected
  demonstration buffer): fully eliminates covering from the final
  greedy policy and recovers substantial true reward, while matching the
  baseline's proxy score.
- **Theoretical result:** for this environment, in any episode,
  `proxy_total = true_total + cost_total` exactly — not an inequality,
  an exact per-episode accounting identity (derivation in the write-up;
  empirically verified against the trained agents in
  `agents/constrained_agent.py::verify_identity`). This means the cost
  signal doesn't just correlate with the proxy/true gap — it exactly
  measures it, which is what justifies the Lagrangian penalty as a
  principled fix rather than a heuristic one.

## Repo structure

```
environments/
  cleaning_env.py        CleaningGridWorld: mechanics, proxy/true rewards, tensor encoding
agents/
  q_network.py            QNetwork (small MLP), ReplayBuffer, epsilon-greedy action selection
  constrained_agent.py     Hand-coded sanity-check policies, potential-based reward shaping,
                           baseline (train_dqn) and constrained (train_dqn_constrained) training
                           loops, greedy rollout + accounting-identity verification
notebooks/
  01_environment_design.ipynb   Working notebook — full build process, diagnostics, and plots
experiments/
  run_baseline_vs_constrained.py   End-to-end script reproducing the comparison from the CLI
results/
  Saved figures (gitignored by default except .gitkeep, add your own)
```

## Running

```bash
pip install -r requirements.txt
python experiments/run_baseline_vs_constrained.py
```

This trains both agents, verifies the accounting identity, prints a
summary table, and saves a comparison figure to `results/`.

## Stated limitations

- Trained on a single fixed grid layout (dirt/bin/start positions
  generated once and reused every episode) rather than generalizing
  across random layouts — a standard simplification for toy gridworld
  demonstrations. Testing generalization is a natural extension.
- The constrained agent fully eliminates the covering exploit but has not
  yet converged to fully optimal navigation within the step budget
  (occasional action oscillation near grid boundaries) — a
  training-efficiency gap distinct from the specification-gaming question
  this project targets.

## Scope

Outer alignment / reward misspecification (specification gaming) with a
constrained-RL mitigation. Explicitly out of scope: robustness, multi-agent
settings.
