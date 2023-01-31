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
        self.wrong_deliveries = []

    def reset(self, **kwargs) -> GymObs:
        self.shaped_rewards = []
        self.sparse_rewards = []
        self.useless_actions = []
        self.wrong_deliveries = []
        return super().reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        observation, reward, done, info = super().step(action)
        self.shaped_rewards.append(info["combined_shaped_r"])
        self.sparse_rewards.append(info["combined_sparse_r"])
        self.useless_actions.append(info["useless_actions"])
        self.wrong_deliveries.append(info["wrong_deliveries"])
        if done:
            info["episode"].update(
                {
                    "ep_shaped_r": sum(self.shaped_rewards),
                    "ep_sparse_r": sum(self.sparse_rewards),
                    "ep_useless_a": sum(self.useless_actions),
                    "ep_wrong_d": sum(self.wrong_deliveries),
                }
            )
        return observation, reward, done, info
