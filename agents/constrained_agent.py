"""
Hand-coded sanity-check policies, potential-based reward shaping,
the proxy-only baseline DQN, and the Lagrangian-penalized constrained
DQN (with a protected, never-evicted demonstration buffer).
"""

import numpy as np
import torch
import torch.nn.functional as F

from environments.cleaning_env import CleaningGridWorld
from agents.q_network import QNetwork, ReplayBuffer, select_action


# ---------------------------------------------------------------------
# Hand-coded policies (Section 3 sanity check, and demo generation)
# ---------------------------------------------------------------------

def nearest_target(agent_pos, grid, target_values):
    """Manhattan-nearest cell whose value is in target_values. None if none exist."""
    candidates = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])
                  if grid[r, c] in target_values]
    if not candidates:
        return None
    ar, ac = agent_pos
    return min(candidates, key=lambda rc: abs(rc[0] - ar) + abs(rc[1] - ac))


def move_toward(agent_pos, target_pos):
    ar, ac = agent_pos
    tr, tc = target_pos
    if ar < tr:
        return CleaningGridWorld.DOWN
    if ar > tr:
        return CleaningGridWorld.UP
    if ac < tc:
        return CleaningGridWorld.RIGHT
    if ac > tc:
        return CleaningGridWorld.LEFT
    return None  # already there


def exploiter_policy(env):
    r, c = env.agent_pos
    if env.grid[r, c] == 1:
        return CleaningGridWorld.COVER
    target = nearest_target(env.agent_pos, env.grid, target_values={1})
    if target is None:
        return CleaningGridWorld.UP
    return move_toward(env.agent_pos, target)


def honest_policy(env):
    r, c = env.agent_pos
    if env.carrying:
        if (r, c) == env.bin_pos:
            return CleaningGridWorld.VACUUM
        return move_toward(env.agent_pos, env.bin_pos)
    if env.grid[r, c] == 1:
        return CleaningGridWorld.VACUUM
    target = nearest_target(env.agent_pos, env.grid, target_values={1})
    if target is None:
        return CleaningGridWorld.UP
    return move_toward(env.agent_pos, target)


def run_policy(policy_fn, n_steps=40, seed=0, grid_size=5, n_dirt=4):
    env = CleaningGridWorld(grid_size=grid_size, n_dirt=n_dirt, seed=seed)
    env.reset()
    proxy_history, true_history = [], []
    cumulative_true = 0
    for _ in range(n_steps):
        action = policy_fn(env)
        _, info = env.step(action)
        cumulative_true += env.true_reward(info)
        proxy_history.append(env.proxy_reward())
        true_history.append(cumulative_true)
    return proxy_history, true_history


# ---------------------------------------------------------------------
# Potential-based reward shaping (Ng, Harada & Russell, 1999)
# F(s,a,s') = gamma*phi(s') - phi(s) leaves the optimal policy unchanged,
# it only provides denser signal to help the agent find it faster.
# ---------------------------------------------------------------------

def potential(env, scale=0.2):
    """phi(s) = -scale * total remaining travel distance, continuous
    across the pickup -> delivery transition (no artificial dip at
    the moment of dropping dirt at the bin)."""
    if env.carrying:
        r, c = env.agent_pos
        br, bc = env.bin_pos
        dist_to_bin = abs(r - br) + abs(c - bc)

        remaining = nearest_target(env.bin_pos, env.grid, target_values={1})
        dist_bin_to_next = 0
        if remaining is not None:
            rr, rc = remaining
            dist_bin_to_next = abs(br - rr) + abs(bc - rc)

        return -scale * (dist_to_bin + dist_bin_to_next)
    else:
        target = nearest_target(env.agent_pos, env.grid, target_values={1})
        if target is None:
            return 0.0
        r, c = env.agent_pos
        tr, tc = target
        return -scale * (abs(r - tr) + abs(c - tc))


def moving_average(x, window=20):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


# ---------------------------------------------------------------------
# Baseline: proxy-only DQN (demonstrates gaming)
# ---------------------------------------------------------------------

def train_dqn(
    n_episodes=500,
    episode_length=40,
    gamma=0.95,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_episodes=300,
    batch_size=64,
    target_update_every=10,
    buffer_capacity=10000,
    shaping_scale=0.2,
    seed=0,
    verbose=True,
):
    env = CleaningGridWorld(grid_size=5, n_dirt=4, seed=seed, fixed_layout=True)
    n_actions = len(CleaningGridWorld.ACTIONS)

    q_network = QNetwork(state_dim=env.state_dim, n_actions=n_actions)
    target_network = QNetwork(state_dim=env.state_dim, n_actions=n_actions)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    episode_proxy_totals, episode_true_totals = [], []

    for ep in range(n_episodes):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * ep / epsilon_decay_episodes)

        env.reset()
        state_vec = env.encode_state()
        prev_proxy = env.proxy_reward()
        ep_proxy_total, ep_true_total = 0, 0

        for t in range(episode_length):
            phi_before = potential(env, scale=shaping_scale)

            action = select_action(q_network, state_vec, epsilon, n_actions)
            _, info = env.step(action)

            new_proxy = env.proxy_reward()
            proxy_delta = new_proxy - prev_proxy
            prev_proxy = new_proxy
            true_r = env.true_reward(info)

            phi_after = potential(env, scale=shaping_scale)
            shaping_term = gamma * phi_after - phi_before
            train_reward = proxy_delta + shaping_term

            next_state_vec = env.encode_state()
            done = (t == episode_length - 1)
            buffer.push(state_vec, action, float(train_reward), next_state_vec, done)

            ep_proxy_total += proxy_delta
            ep_true_total += true_r
            state_vec = next_state_vec

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(dim=1)[0]
                    targets = rewards + gamma * next_q_values * (1 - dones)
                loss = F.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        episode_proxy_totals.append(ep_proxy_total)
        episode_true_totals.append(ep_true_total)

        if ep % target_update_every == 0:
            target_network.load_state_dict(q_network.state_dict())

        if verbose and ep % 50 == 0:
            print(f"Episode {ep:4d} | epsilon {epsilon:.2f} | "
                  f"proxy {ep_proxy_total:5.1f} | true {ep_true_total:.1f}")

    return q_network, episode_proxy_totals, episode_true_totals


# ---------------------------------------------------------------------
# Constrained fix: fixed-lambda Lagrangian penalty + protected demo buffer
# ---------------------------------------------------------------------

def train_dqn_constrained(
    n_episodes=800,
    episode_length=40,
    gamma=0.95,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_episodes=300,
    batch_size=64,
    target_update_every=10,
    buffer_capacity=10000,
    shaping_scale=0.2,
    lambda_coef=2.0,
    n_demo_episodes=20,
    demo_frac=0.5,
    seed=0,
    verbose=True,
):
    env = CleaningGridWorld(grid_size=5, n_dirt=4, seed=seed, fixed_layout=True)
    n_actions = len(CleaningGridWorld.ACTIONS)

    q_network = QNetwork(state_dim=env.state_dim, n_actions=n_actions)
    target_network = QNetwork(state_dim=env.state_dim, n_actions=n_actions)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    demo_buffer = ReplayBuffer(capacity=n_demo_episodes * episode_length + 10)
    for _ in range(n_demo_episodes):
        env.reset()
        state_vec = env.encode_state()
        prev_proxy = env.proxy_reward()
        for t in range(episode_length):
            phi_before = potential(env, scale=shaping_scale)
            action = honest_policy(env)
            _, info = env.step(action)

            new_proxy = env.proxy_reward()
            proxy_delta = new_proxy - prev_proxy
            prev_proxy = new_proxy

            cost_t = 1.0 if info.get("covered") else 0.0
            if t == episode_length - 1 and env.carrying:
                cost_t += 1.0

            phi_after = potential(env, scale=shaping_scale)
            shaping_term = gamma * phi_after - phi_before
            train_reward = proxy_delta + shaping_term - lambda_coef * cost_t

            next_state_vec = env.encode_state()
            done = (t == episode_length - 1)
            demo_buffer.push(state_vec, action, float(train_reward), next_state_vec, done)
            state_vec = next_state_vec

    episode_proxy_totals, episode_true_totals, episode_cost_totals = [], [], []

    for ep in range(n_episodes):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * ep / epsilon_decay_episodes)

        env.reset()
        state_vec = env.encode_state()
        prev_proxy = env.proxy_reward()
        ep_proxy_total, ep_true_total, ep_cost_total = 0, 0, 0

        for t in range(episode_length):
            phi_before = potential(env, scale=shaping_scale)

            action = select_action(q_network, state_vec, epsilon, n_actions)
            _, info = env.step(action)

            new_proxy = env.proxy_reward()
            proxy_delta = new_proxy - prev_proxy
            prev_proxy = new_proxy
            true_r = env.true_reward(info)

            cost_t = 1.0 if info.get("covered") else 0.0
            if t == episode_length - 1 and env.carrying:
                cost_t += 1.0

            phi_after = potential(env, scale=shaping_scale)
            shaping_term = gamma * phi_after - phi_before
            train_reward = proxy_delta + shaping_term - lambda_coef * cost_t

            next_state_vec = env.encode_state()
            done = (t == episode_length - 1)
            buffer.push(state_vec, action, float(train_reward), next_state_vec, done)

            ep_proxy_total += proxy_delta
            ep_true_total += true_r
            ep_cost_total += cost_t
            state_vec = next_state_vec

            if len(buffer) >= batch_size:
                n_demo = int(batch_size * demo_frac)
                n_regular = batch_size - n_demo

                s1, a1, r1, ns1, d1 = buffer.sample(n_regular)
                s2, a2, r2, ns2, d2 = demo_buffer.sample(n_demo)

                states = torch.cat([s1, s2])
                actions = torch.cat([a1, a2])
                rewards = torch.cat([r1, r2])
                next_states = torch.cat([ns1, ns2])
                dones = torch.cat([d1, d2])

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(dim=1)[0]
                    targets = rewards + gamma * next_q_values * (1 - dones)
                loss = F.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        episode_proxy_totals.append(ep_proxy_total)
        episode_true_totals.append(ep_true_total)
        episode_cost_totals.append(ep_cost_total)

        if ep % target_update_every == 0:
            target_network.load_state_dict(q_network.state_dict())

        if verbose and ep % 50 == 0:
            print(f"Episode {ep:4d} | epsilon {epsilon:.2f} | "
                  f"proxy {ep_proxy_total:5.1f} | true {ep_true_total:.1f} | cost {ep_cost_total:.1f}")

    return q_network, episode_proxy_totals, episode_true_totals, episode_cost_totals


# ---------------------------------------------------------------------
# Evaluation: greedy rollout + the proxy = true + cost identity check
# ---------------------------------------------------------------------

def rollout_greedy(q_network, env, episode_length=40):
    env.reset()
    state_vec = env.encode_state()
    action_log = []
    proxy_history = [env.proxy_reward()]
    true_total = 0
    proxy_delta_total = 0

    prev_proxy = env.proxy_reward()
    for t in range(episode_length):
        with torch.no_grad():
            action = int(torch.argmax(q_network(state_vec.unsqueeze(0)), dim=1).item())
        _, info = env.step(action)

        new_proxy = env.proxy_reward()
        proxy_delta_total += (new_proxy - prev_proxy)
        prev_proxy = new_proxy

        true_total += env.true_reward(info)
        action_log.append(CleaningGridWorld.ACTIONS[action])
        proxy_history.append(new_proxy)
        state_vec = env.encode_state()

    return action_log, proxy_history, true_total, proxy_delta_total


def verify_identity(actions_list, true_total, proxy_delta_total, env_final):
    """
    Checks the exact per-episode accounting identity:
        proxy_total = true_total + cost_total
    """
    cover_count = actions_list.count("cover")
    abandoned = 1 if env_final.carrying else 0
    cost_total = cover_count + abandoned
    holds = (proxy_delta_total == true_total + cost_total)
    print(f"proxy_total (net) = {proxy_delta_total}, true_total = {true_total}, cost_total = {cost_total}")
    print(f"true_total + cost_total = {true_total + cost_total}  (should equal proxy_total)")
    print(f"Identity holds: {holds}\n")
    return holds