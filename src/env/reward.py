"""Reward components (priority order):

  OOM PENALTY  (limit < usage)
    Range: [-4, -2].

  COLD-START GUARD  (init phase: idle_steps_norm > 0 AND cpu_usage < 0.2)
    The agent is in the initialization phase — workload has not started yet.
    Return neutral reward (0.0) so there is NO incentive to trim.
    Only the trend bonus and failure penalty are still applied.

  EFFICIENCY REWARD  (active phase only: episode_peak_norm > 0)
    Piecewise-linear around target_util = 1 / (1 + margin) ≈ 0.909:
      - util = 0   -> reward = 0  (pure over-provision, but workload is active)
      - util = target -> reward = 1.0  (only positive region)
      - util -> 1.0 -> reward -> 0  (steep decline toward OOM)

  TREND ALIGNMENT BONUS
    Normal   : usage_trend ∈ (0.10, 0.05]   -> +0.10 for expand
    Big spike : usage_trend > 0.05           -> +0.30 (init->active transition)
    Trim      : usage_trend < −0.10          -> +0.05 for trim

  FAILURE PRESSURE PENALTY
    -0.20 flat penalty for trimming when failure rate is above threshold.

  INIT TRIM PENALTY
    range from -0.15 to -0.75 per trim action during init phase.
    Discourages trimming before workload starts,
    complementing the terminal-OOM mechanism in real_data_env.py.

Output clipped to [-9.0, +1.5]
-9.0 just as a safe lower bound to prevent runaway negative rewards in case of bugs/exploits.
"""


def compute_real_data_reward(
    memory_usage: float,
    memory_upper_limit: float,
    target_margin_pct: float = 0.10,
    memory_failures_norm: float = 0.0,
    usage_trend: float = 0.0,
    action: float = 0.0,
    failure_rate_threshold: float = 0.5,
    idle_steps_norm: float = 0.0,
    episode_peak_norm: float = 0.0,
    cpu_usage: float = 0.0,
) -> float:

    # Compoennt 1: OOM penalty (highest priority, always applies)
    # OOM penalty (highest priority, always applies)
    if memory_upper_limit < memory_usage:
        deficit_ratio = (memory_usage - memory_upper_limit) / max(memory_usage, 1.0)
        penalty = -2.0 - 2.0 * min(deficit_ratio, 1.0)   # ∈ [-4, -2]
        return float(penalty)

    # Component 2: Cold start guard
    in_init_phase = (idle_steps_norm > 0.0 and cpu_usage < 0.2)
    if in_init_phase:
        # Neutral reward: no efficiency penalty for over-provisioning during init.
        # The agent should HOLD the limit, not trim it.
        # We still apply trend bonus (for the rare case usage starts rising) and
        # failure penalty (discourage trimming if failures fire during init).
        reward = 0.0

        # Trend alignment still applies (proactive expand if usage starts rising)
        if usage_trend > 0.2 and action > 0.0:
            reward += 0.3 * min(usage_trend, 1.0)
        elif usage_trend > 0.1 and action > 0.0:
            reward += 0.1 * min(usage_trend, 1.0)
        elif action > 0.0:
            reward -= 0.15

        if memory_failures_norm > failure_rate_threshold and action < 0.0:
            reward -= 0.2

        # Component 6
        if action < 0.0:
            reward -= 0.15 # Y ∈ [-0.15,-0.25,-0.50,-0.75]

        return float(max(-9.0, min(1.5, reward)))
        

    # Component 3
    # Main efficiency reward
    # Apply normal efficiency shaping.
    util_ratio  = memory_usage / max(memory_upper_limit, 1.0)
    target_util = 1.0 / (1.0 + target_margin_pct)  # ≈ 0.909 at margin=0.10
    gap = util_ratio - target_util
    if gap <= 0.0:
        efficiency = 1.0 + gap / target_util   # ∈ [0, 1]
    else:
        efficiency = max(0.0, 1.0 - 10.0 * gap)   # ∈ [0, 1)
    reward = efficiency   # ∈ [0, 1]
    
    # Component 4: Trend alignment bonus
    if usage_trend > 0.2 and action > 0.0:
        # Large spike during active phase
        reward += 0.3 * min(usage_trend, 1.0)   # up to +0.30
    elif usage_trend > 0.1 and action > 0.0:
        reward += 0.1 * min(usage_trend, 1.0)   # up to +0.10
    elif usage_trend < -0.1 and action < 0.0:
        reward += 0.05 * min(-usage_trend, 1.0) # up to +0.05

    # Component 5: Failure pressure penalty still applies during init phase to discourage trimming if failures are firing.
    if memory_failures_norm > failure_rate_threshold and action < 0.0:
        reward -= 0.2

    return float(max(-9.0, min(1.5, reward)))
