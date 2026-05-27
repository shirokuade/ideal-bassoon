"""Gymnasium environment that replays HPC workload traces for PPO training.

Observation space (14D):
    [0]  memory_working_set   - working_set / original_limit
    [1]  memory_rss           - rss / original_limit
    [2]  memory_usage         - total usage / original_limit
    [3]  memory_cache         - cache / original_limit
    [4]  memory_limit_kube    - K8s limit / original_limit
    [5]  memory_request_kube  - K8s request / original_limit
    [6]  memory_failures_rate - clip(log1p(failures/working_set * 1e8) / 10, 0, 1)
    [7]  agent_limit          - agent's current limit / original_limit
    [8]  usage_trend          - diff(working_set) / original_limit
    [9]  utilization_ratio    - usage / agent_limit
    [10] cpu_limit_kube       - raw cpu_limit_kube in cores
    [11] cpu_usage_rate       - cpu_usage_rate / cpu_limit_kube (clipped 0-2)
    ── Init-phase features ───────────────────────────────────────────────────
    [12] idle_steps_norm      - consecutive cold steps / IDLE_NORM, clipped [0, 1]
                                Counts steps where working_set < COLD_USAGE_PCT × orig.
                                Grows during init, resets to 0 the moment usage spikes.
    [13] episode_peak_norm    - max(working_set seen so far) / original_limit
                                = 0.0 for the entire init phase.
                                Jumps at the first spike and never decreases.
                                Unambiguous signal: peak=0 ↔ workload not yet started.

Action space: 1D continuous [-1, 1]
    [0]  direction + magnitude
           positive = expand limit, negative = trim limit
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.agent.config import Config
from src.data.loader import PodTrace, load_dataset
from src.env.reward import compute_real_data_reward


class RealDataMemoryEnv(gym.Env):
    """RL environment replaying real Kubernetes pod memory traces."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Config | None = None,
        pod_traces: list[PodTrace] | None = None,
        workload_filter: str | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config or Config()
        self.render_mode = render_mode

        if pod_traces is not None:
            self.pod_traces = pod_traces
        else:
            all_traces = load_dataset(self.config.DATASET_DIR)
            self.pod_traces = all_traces

        if workload_filter:
            self.pod_traces = [
                t for t in self.pod_traces if t.workload_type == workload_filter
            ]

        if not self.pod_traces:
            raise ValueError("No pod traces available after filtering.")

        # Observation space: 14D
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.5, 1.5, 1.5, 1.0, 1.5, 1.5, 1.0, 1.5, 1.0, 2.0, 100.0, 2.0, 1.0, 1.5],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Action space: 1D continuous [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Internal state (initialized in reset)
        self.current_trace: PodTrace | None = None
        self.current_step = 0
        self.memory_usage = 0.0
        self.memory_upper_limit = 0.0
        self.prev_usage = 0.0
        self.original_limit = 0.0
        self.original_cpu_limit = 0.0
        self.current_failures_raw: float = 0.0
        self.memory_failures_norm: float = 0.0
        self.consecutive_idle_steps: int = 0
        self.episode_peak_usage: float = 0.0
        self.episode_history: list[dict] = []

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        idx = self.np_random.integers(0, len(self.pod_traces))
        self.current_trace = self.pod_traces[idx]
        self.current_step = 0

        row = self.current_trace.data.iloc[0]
        self.memory_usage = self._safe_get(row, "memory_working_set", 100.0)
        initial_failures_raw = self._safe_get(row, "memory_failures_rate", 0.0)
        self.current_failures_raw = initial_failures_raw
        self.memory_failures_norm = 0.0
        self.prev_usage = self.memory_usage

        self.original_limit = self._find_original_limit()
        self.memory_upper_limit = self.original_limit
        self.original_cpu_limit = self._safe_get(
            self.current_trace.data.iloc[0], "cpu_limit_kube", 1.0
        )

        # Init-phase counters start at zero; step() updates them each row.
        self.consecutive_idle_steps = 0
        self.episode_peak_usage = 0.0

        self.episode_history = []

        obs = self._get_observation(row)
        info = {
            "memory_usage": self.memory_usage,
            "memory_upper_limit": self.memory_upper_limit,
            "original_limit": self.original_limit,
            "pod_name": self.current_trace.pod_name,
            "workload_type": self.current_trace.workload_type,
            "pod_succeeded": self.current_trace.succeeded,
            "memory_failures_rate": initial_failures_raw,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_0 = float(action[0])  # direction + magnitude ∈ [-1, 1]

        adjustment = action_0 * self.config.MAX_ADJUSTMENT_PCT * self.memory_upper_limit
        self.memory_upper_limit = float(np.clip(
            self.memory_upper_limit + adjustment,
            self.config.MIN_MEMORY_LIMIT,
            self.original_limit * 2.0,
        ))

        # Advance to next row
        self.current_step += 1
        self.prev_usage = self.memory_usage

        max_steps = len(self.current_trace.data)
        if self.current_step >= max_steps:
            truncated = True
            row = self.current_trace.data.iloc[-1]
        else:
            truncated = False
            row = self.current_trace.data.iloc[self.current_step]

        self.memory_usage = self._safe_get(row, "memory_working_set", self.prev_usage)

        norm = max(self.original_limit, 1.0)
        obs_8_trend = float(np.clip(
            (self.memory_usage - self.prev_usage) / norm, -2.0, 2.0
        ))

        # Update init-phase counters.
        # episode_peak only tracks ACTIVE usage (above cold threshold) so it
        # stays at exactly 0.0 during the entire init phase, despite the tiny
        # OS baseline memory that is always present.
        if (self.memory_usage / norm) >= self.config.COLD_USAGE_PCT:
            self.episode_peak_usage = max(self.episode_peak_usage, self.memory_usage)

        if (self.memory_usage / norm) < self.config.COLD_USAGE_PCT:
            self.consecutive_idle_steps += 1
        else:
            self.consecutive_idle_steps = 0

        # Update failure state (used by next step's action scaling)
        failures_raw = self._safe_get(row, "memory_failures_rate", 0.0)
        self.current_failures_raw = failures_raw
        _ratio_new = max(failures_raw, 0.0) / max(self.memory_usage, 1.0)
        self.memory_failures_norm = float(np.clip(np.log1p(_ratio_new * 1e8) / 10.0, 0.0, 1.0))

        idle_steps_norm   = float(np.clip(self.consecutive_idle_steps / self.config.IDLE_NORM, 0.0, 1.0))
        episode_peak_norm = float(np.clip(self.episode_peak_usage / norm, 0.0, 1.5))

        cpu_limit_val = self._safe_get(row, "cpu_limit_kube", self.original_cpu_limit)
        cpu_usage_rate = self._safe_get(row, "cpu_usage_rate", 0.0)
        cpu_usage = float(np.clip(
            cpu_usage_rate / max(cpu_limit_val, 0.001),
            0.0, 2.0,
        ))

        reward = compute_real_data_reward(
            memory_usage=self.memory_usage,
            memory_upper_limit=self.memory_upper_limit,
            target_margin_pct=self.config.TARGET_MARGIN_PCT,
            memory_failures_norm=self.memory_failures_norm,
            usage_trend=obs_8_trend,
            action=action_0,
            failure_rate_threshold=self.config.FAILURE_RATE_THRESHOLD,
            idle_steps_norm=idle_steps_norm,
            episode_peak_norm=episode_peak_norm,
            cpu_usage=cpu_usage,
        )

        # Terminal OOM
        # If OOM occurs while the workload is active (episode_peak_norm > 0),
        # the pod would be OOM-killed in production -> terminate the episode.
        # This removes the "trim - OOM - recover - high efficiency" exploit:
        # with no future rewards after OOM, the strategy becomes unprofitable.

        oom = self.memory_upper_limit < self.memory_usage
        terminated = bool(oom and episode_peak_norm > 0.0)

        agent_waste = max(0.0, self.memory_upper_limit - self.memory_usage)
        original_waste = max(0.0, self.original_limit - self.memory_usage)
        self.episode_history.append(
            {
                "step": self.current_step,
                "usage": self.memory_usage,
                "agent_limit": self.memory_upper_limit,
                "original_limit": self.original_limit,
                "reward": reward,
                "agent_waste": agent_waste,
                "original_waste": original_waste,
                "oom": oom,
                "terminated": terminated,
            }
        )

        obs = self._get_observation(row)
        info = {
            "memory_usage": self.memory_usage,
            "memory_upper_limit": self.memory_upper_limit,
            "waste": agent_waste,
            "original_waste": original_waste,
            "oom": oom,
            "terminated_oom": terminated,
            "memory_failures_rate": failures_raw,
        }
        return obs, float(reward), terminated, truncated, info

    def _get_observation(self, row) -> np.ndarray:
        # Construct the 14D normalized observation from a DataFrame row
        norm = max(self.original_limit, 1.0)

        working_set = self.memory_usage
        obs_0 = np.clip(working_set / norm, 0.0, 2.0)

        rss = self._safe_get(row, "memory_rss", working_set)
        obs_1 = np.clip(rss / norm, 0.0, 2.0)

        mem_usage_total = self._safe_get(row, "memory_usage", working_set)
        obs_2 = np.clip(mem_usage_total / norm, 0.0, 2.0)

        cache = self._safe_get(row, "memory_cache", 0.0)
        obs_3 = float(np.clip(cache / norm, 0.0, 1.0))

        limit_kube = self._safe_get(row, "memory_limit_kube", self.original_limit)
        obs_4 = np.clip(limit_kube / norm, 0.0, 2.0)

        request_kube = self._safe_get(row, "memory_request_kube", limit_kube)
        obs_5 = np.clip(request_kube / norm, 0.0, 2.0)

        failures = self._safe_get(row, "memory_failures_rate", 0.0)
        _f_ratio = max(failures, 0.0) / max(working_set, 1.0)
        obs_6 = float(np.clip(np.log1p(_f_ratio * 1e8) / 10.0, 0.0, 1.0))

        obs_7 = np.clip(self.memory_upper_limit / norm, 0.0, 2.0)

        obs_8 = np.clip((self.memory_usage - self.prev_usage) / norm, -1.0, 1.0)

        obs_9 = self.memory_usage / max(self.memory_upper_limit, 1.0)

        cpu_limit = self._safe_get(row, "cpu_limit_kube", self.original_cpu_limit)
        obs_10 = np.clip(cpu_limit / max(self.original_cpu_limit, 0.001), 0.0, 2.0)

        cpu_usage = self._safe_get(row, "cpu_usage_rate", 0.0)
        obs_11 = np.clip(
            cpu_usage / cpu_limit if cpu_limit > 0.0 else 0.0,
            0.0,
            2.0,
        )

        obs_12 = float(np.clip(self.consecutive_idle_steps / self.config.IDLE_NORM, 0.0, 1.0))

        obs_13 = float(np.clip(self.episode_peak_usage / norm, 0.0, 1.5))

        return np.array(
            [obs_0, obs_1, obs_2, obs_3, obs_4, obs_5,
             obs_6, obs_7, obs_8, obs_9, obs_10, obs_11,
             obs_12, obs_13],
            dtype=np.float32,
        )

    def _find_original_limit(self) -> float:
        df = self.current_trace.data

        if "memory_limit_kube" in df.columns:
            valid = df["memory_limit_kube"].dropna()
            if not valid.empty:
                return float(valid.iloc[0])

        if "memory_working_set" in df.columns:
            peak = df["memory_working_set"].dropna().max()
            if not np.isnan(peak) and peak > 0:
                return float(peak * 2.0)

        return self.memory_usage * 2.0

    @staticmethod
    def _safe_get(row, column: str, default: float = 0.0) -> float:
        if column in row.index:
            val = row[column]
            if not np.isnan(val):
                return float(val)
        return default
