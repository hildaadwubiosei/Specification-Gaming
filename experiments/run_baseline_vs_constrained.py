"""
Reproduces the core result: a proxy-only DQN learns to game the proxy
reward in CleaningGridWorld; a Lagrangian-penalized constrained DQN
(with a protected demonstration buffer) substantially closes the gap.
Also verifies the exact accounting identity proxy_total = true_total +
cost_total on both agents' final greedy policies.

Run from the repo root:
    python experiments/run_baseline_vs_constrained.py
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.cleaning_env import CleaningGridWorld
from agents.constrained_agent import (
    train_dqn,
    train_dqn_constrained,
    rollout_greedy,
    verify_identity,
    moving_average,
)

SEED = 0
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("Training baseline (proxy-only) DQN...")
    print("=" * 60)
    baseline_q_net, baseline_proxy_log, baseline_true_log = train_dqn(
        n_episodes=500, seed=SEED
    )

    print()
    print("=" * 60)
    print("Training constrained (Lagrangian-penalized) DQN...")
    print("=" * 60)
    constrained_q_net, constrained_proxy_log, constrained_true_log, constrained_cost_log = \
        train_dqn_constrained(n_episodes=800, lambda_coef=2.0, demo_frac=0.5, seed=SEED)

    # --- greedy rollouts for both final policies ---
    eval_env_b = CleaningGridWorld(grid_size=5, n_dirt=4, seed=SEED, fixed_layout=True)
    eval_env_b.reset()
    actions_b, proxy_hist_b, true_total_b, proxy_delta_b = rollout_greedy(baseline_q_net, eval_env_b)

    eval_env_c = CleaningGridWorld(grid_size=5, n_dirt=4, seed=SEED, fixed_layout=True)
    eval_env_c.reset()
    actions_c, proxy_hist_c, true_total_c, proxy_delta_c = rollout_greedy(constrained_q_net, eval_env_c)

    # --- verify the exact accounting identity on both ---
    print()
    print("=" * 60)
    print("Verifying identity: proxy_total = true_total + cost_total")
    print("=" * 60)
    print("Baseline:")
    verify_identity(actions_b, true_total_b, proxy_delta_b, eval_env_b)
    print("Constrained:")
    verify_identity(actions_c, true_total_c, proxy_delta_c, eval_env_c)

    # --- summary table ---
    print("=" * 60)
    print(f"{'Metric':<15}{'Baseline':>12}{'Constrained':>14}")
    print(f"{'Proxy reward':<15}{proxy_hist_b[-1]:>12}{proxy_hist_c[-1]:>14}")
    print(f"{'True reward':<15}{true_total_b:>12}{true_total_c:>14}")
    print(f"{'Cover actions':<15}{actions_b.count('cover'):>12}{actions_c.count('cover'):>14}")
    print("=" * 60)

    # --- figure: training curves + final rollout comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(moving_average(baseline_true_log), color="tomato", label="Baseline (proxy-only)")
    axes[0].plot(moving_average(constrained_true_log), color="steelblue", label="Constrained (Lagrangian)")
    axes[0].set_title("True reward over training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("True reward (moving avg, window=20)")
    axes[0].legend()

    labels = ["Proxy", "True", "Cover actions"]
    baseline_vals = [proxy_hist_b[-1], true_total_b, actions_b.count("cover")]
    constrained_vals = [proxy_hist_c[-1], true_total_c, actions_c.count("cover")]

    x = np.arange(len(labels))
    width = 0.35
    axes[1].bar(x - width / 2, baseline_vals, width, label="Baseline", color="tomato")
    axes[1].bar(x + width / 2, constrained_vals, width, label="Constrained", color="steelblue")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Final greedy-policy rollout")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "baseline_vs_constrained.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
