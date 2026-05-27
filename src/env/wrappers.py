

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FrameStackWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, n_frames: int):
        super().__init__(env)
        assert n_frames > 1, "n_frames must be > 1"
        self.n_frames = n_frames
        self._frames: deque = deque(maxlen=n_frames)

        obs_dim = env.observation_space.shape[0]
        low = np.tile(env.observation_space.low, n_frames)
        high = np.tile(env.observation_space.high, n_frames)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self.observation(obs), reward, terminated, truncated, info
