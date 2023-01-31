from typing import Union

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class OvercookedMonitor(Monitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shaped_rewards = []
        self.sparse_rewards = []
        self.useless_actions = []

    def reset(self, **kwargs) -> GymObs:
        self.shaped_rewards = []
        self.sparse_rewards = []
        self.useless_actions = []
        return super().reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        observation, reward, done, info = super().step(action)
        self.shaped_rewards.append(info["combined_shaped_r"])
        self.sparse_rewards.append(info["combined_sparse_r"])
        self.useless_actions.append(info["useless_actions"])
        if done:
            ep_shaped_r = sum(self.shaped_rewards)
            ep_sparse_r = sum(self.sparse_rewards)
            ep_useless_a = sum(self.useless_actions)
            info["episode"].update(
                {
                    "ep_shaped_r": ep_shaped_r,
                    "ep_sparse_r": ep_sparse_r,
                    "ep_useless_a": ep_useless_a,
                }
            )
        return observation, reward, done, info
