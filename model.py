"""Spatial public goods game with group-event Q-learning.

Actions: 0 = D, 1 = C. Joint states: CG, CB, DG, DB.
Only NumPy and Numba are required; no plotting or paper data are used.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def seed_events(seed):
    """Seed Numba's RNG separately from lattice initialization."""
    np.random.seed(seed)


@njit(cache=True)
def state_index(action, good, representation):
    if representation == 1:  # behavior-only: C, D
        return 0 if action == 1 else 1
    if representation == 2:  # environment-only: G, B
        return 0 if good else 1
    return (0 if action == 1 else 2) + (0 if good else 1)


@njit(cache=True)
def penalty(action, nc, threshold, beta, independent):
    """Return a nonnegative penalty magnitude; reward subtracts it."""
    if independent:
        return beta * (5 - nc) if action == 1 else beta * nc
    if nc >= threshold and action == 0:
        return beta * nc
    if nc < threshold and action == 1:
        return beta * (5 - nc)
    return 0.0


@njit(cache=True)
def advance(actions, q, r, threshold, beta, alpha, gamma, epsilon,
            independent, representation, mcs):
    """Advance a bounded chunk. Row 0 describes its first completed MCS.

    Visit counts include every group member's decision, including repeat visits.
    Advantages are measured before the corresponding Q updates. Outputs record
    each MCS, without temporal binning. Missing-state advantages are handled by
    the caller. Chunk boundaries do not reset the random stream or Q tables.
    """
    length = actions.shape[0]
    states = q.shape[2]
    rho = np.empty(mcs)
    counts = np.zeros((mcs, states), dtype=np.int64)
    advantage_sums = np.zeros((mcs, states))
    group = np.empty((5, 2), dtype=np.int64)
    old_states = np.empty(5, dtype=np.int8)
    choices = np.empty(5, dtype=np.int8)
    for t in range(mcs):
        for event in range(length * length):
            i, j = np.random.randint(0, length), np.random.randint(0, length)
            group[0, 0], group[0, 1] = i, j
            group[1, 0], group[1, 1] = (i - 1) % length, j
            group[2, 0], group[2, 1] = (i + 1) % length, j
            group[3, 0], group[3, 1] = i, (j - 1) % length
            group[4, 0], group[4, 1] = i, (j + 1) % length
            nc_old = 0
            for k in range(5):
                nc_old += actions[group[k, 0], group[k, 1]]
            # All decisions use the old group; do not write actions in this loop.
            for k in range(5):
                x, y = group[k, 0], group[k, 1]
                s = state_index(actions[x, y], nc_old >= threshold, representation)
                old_states[k] = s
                qd, qc = q[x, y, s, 0], q[x, y, s, 1]
                counts[t, s] += 1
                advantage_sums[t, s] += qc - qd
                if np.random.random() < epsilon:
                    choices[k] = np.random.randint(0, 2)
                elif qd == qc:
                    choices[k] = np.random.randint(0, 2)
                else:
                    choices[k] = 0 if qd > qc else 1
            nc_new = 0
            for k in range(5):
                actions[group[k, 0], group[k, 1]] = choices[k]
                nc_new += choices[k]
            # Reward and next state refer to the SAME post-action group.
            for k in range(5):
                x, y = group[k, 0], group[k, 1]
                s, a = old_states[k], choices[k]
                next_s = state_index(a, nc_new >= threshold, representation)
                cost = penalty(a, nc_new, threshold, beta, independent)
                reward = r * nc_new / 5.0 - a - cost
                future = gamma * max(q[x, y, next_s, 0], q[x, y, next_s, 1])
                q[x, y, s, a] += alpha * (reward + future - q[x, y, s, a])
        rho[t] = np.mean(actions)
    return rho, counts, advantage_sums
