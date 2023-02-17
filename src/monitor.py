from typing import Any, Union

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class OvercookedMonitor(Monitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shaped_rewards = []
        self.sparse_rewards = []
        self.punishments = []
        self.useless_actions = []
        self.wrong_deliveries = []
        self.collisions = []

    def reset(self, **kwargs) -> GymObs:
        self.shaped_rewards = []
        self.sparse_rewards = []
        self.punishments = []
        self.useless_actions = []
        self.wrong_deliveries = []
        self.collisions = []
        return super().reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        observation, reward, done, info = super().step(action)
        self.shaped_rewards.append(info["combined_shaped_r"])
        self.punishments.append(info["combined_punishment"])
        self.sparse_rewards.append(info["combined_sparse_r"])
        self.useless_actions.append(info["useless_actions"])
        self.wrong_deliveries.append(info["wrong_deliveries"])
        self.collisions.append(info["collision"])
        if done:
            info["episode"].update(
                {
                    "ep_shaped_r": sum(self.shaped_rewards),
                    "ep_sparse_r": sum(self.sparse_rewards),
                    "ep_punishment": sum(self.punishments),
                    "ep_useless_a": sum(self.useless_actions),
                    "ep_wrong_d": sum(self.wrong_deliveries),
                    "ep_collisions": sum(self.collisions),
                }
            )
        return observation, reward, done, info

    # TODO: This is hacky and ugly, replace this whole monitor with VecMonitor to allow unwrapping and setting the value directly in the environment object
    @property
    def shaped_reward_coef(self):
        return self._learning_rate_coef

    @shaped_reward_coef.setter
    def shaped_reward_coef(self, value: float):
        assert 0.0 <= value and value <= 1.0
        self.env.unwrapped.shaped_reward_coef = value

    @property
    def punishment_coef(self):
        return self._punishment_coef

    @punishment_coef.setter
    def punishment_coef(self, value: float):
        assert 0.0 <= value and value <= 1.0
        self.env.unwrapped.punishment_coef = value
