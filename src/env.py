from collections.abc import Callable
from typing import Optional

import gym
import numpy as np

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import (EVENT_TYPES,
                                                 OvercookedGridworld)


class Overcooked(gym.Env):
    env_name = "Overcooked-v1"

    def __init__(
        self,
        mdp: OvercookedGridworld,
        featurize_fn: Optional[Callable] = None,
        start_state_fn: Optional[Callable] = None,
        horizon: int = 500,
    ):
        self.mdp = mdp
        self.featurize_fn = (
            featurize_fn
            if featurize_fn is not None
            else self.mdp.lossless_state_encoding
        )
        self.start_state_fn = (
            start_state_fn
            if start_state_fn is not None
            else self.mdp.get_standard_start_state
        )
        self.horizon = horizon
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.MultiDiscrete(
            [len(Action.ALL_ACTIONS), len(Action.ALL_ACTIONS)]
        )
        self.reset()

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        # TODO: currently only the lossless state encoding works (multiple binary maps)
        obs_shape = self.featurize_fn(dummy_state).shape
        return gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.uint8)

    def reset(self):
        self.mdp_state = self.start_state_fn()
        obs = self.featurize_fn(self.mdp_state)

        events_dict = {
            k: [[] for _ in range(self.mdp.num_players)] for k in EVENT_TYPES
        }
        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array(
                [0.0] * self.mdp.num_players
            ),
            "cumulative_shaped_rewards_by_agent": np.array(
                [0.0] * self.mdp.num_players
            ),
            "cumulative_useless_actions_by_agent": np.array([0] * self.mdp.num_players),
        }

        self.game_stats = {**events_dict, **rewards_dict}
        return obs

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        next_state, mdp_infos = self.mdp.get_state_transition(
            self.mdp_state, joint_action
        )
        self.mdp_state = next_state
        done = self._is_done()
        obs = self.featurize_fn(self.mdp_state)
        info = self._prepare_info_dict(mdp_infos)
        reward = info["combined_sparse_r"] + info["combined_shaped_r"]

        return obs, reward, done, info

    def _prepare_info_dict(self, mdp_infos):
        env_info = {
            "useless_actions": sum(mdp_infos["useless_actions_by_agent"]),
            "wrong_deliveries": sum(mdp_infos["wrong_deliveries_by_agent"]),
            "combined_sparse_r": sum(mdp_infos["sparse_reward_by_agent"]),
            "combined_shaped_r": sum(mdp_infos["shaped_reward_by_agent"]),
        }
        return env_info

    def _is_done(self):
        return self.mdp_state.timestep >= self.horizon or self.mdp.is_terminal(
            self.mdp_state
        )
