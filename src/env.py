from collections.abc import Callable
from typing import Optional

import gym
import numpy as np
from gym.error import DependencyNotInstalled

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


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
        self.screen = None
        self.state_visualizer = None
        self.shaped_reward_coef = 1.0
        self.punishment_coef = 0.0
        self.reset()

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        # TODO: currently only the lossless state encoding works (multiple binary maps)
        obs_shape = self.featurize_fn(dummy_state).shape
        return gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.uint8)

    def reset(self):
        self.mdp_state = self.start_state_fn()
        # keep track of score for easier visualization
        self.score = 0
        self.collisions = 0
        self.useless_actions = 0
        obs = self.featurize_fn(self.mdp_state)
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
        reward = (
            info["combined_sparse_r"]
            + self.shaped_reward_coef * info["combined_shaped_r"]
            + self.punishment_coef * info["combined_punishment"]
        )
        self.score += info["combined_sparse_r"]
        self.collisions += info["collision"]
        self.useless_actions += info["useless_actions"]

        return obs, reward, done, info

    def render(self, mode="human"):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled("pygame is not installed")

        if self.state_visualizer is None:
            self.state_visualizer = StateVisualizer()

        grid = self.mdp.mdp_params["terrain"]
        hud_data = {
            "timestep": self.mdp_state.timestep,
            "all_orders": [r.to_dict() for r in self.mdp_state.all_orders],
            # "bonus_orders": [r.to_dict() for r in self.mdp_state.bonus_orders],
            "score": self.score,
            "collisions": self.collisions,
            "useless_actions": self.useless_actions,
        }

        surface = self.state_visualizer.render_state(self.mdp_state, grid, hud_data)
        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (surface.get_width(), surface.get_height())
                )
            elif mode == "rgb_array":
                self.screen = pygame.Surface(
                    (surface.get_width(), surface.get_height())
                )
            else:
                raise NotImplementedError

        self.screen.blit(surface, (0, 0))
        if mode == "human":
            pygame.display.flip()

        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(2, 1, 0)
            )

    def _prepare_info_dict(self, mdp_infos):
        env_info = {
            "useless_actions": sum(mdp_infos["useless_actions_by_agent"]),
            "wrong_deliveries": sum(mdp_infos["wrong_deliveries_by_agent"]),
            "combined_sparse_r": sum(mdp_infos["sparse_reward_by_agent"]),
            "combined_shaped_r": sum(mdp_infos["shaped_reward_by_agent"]),
            "combined_punishment": sum(mdp_infos["punishment_by_agent"]),
            "collision": mdp_infos["collision"],
            "punishment_coef": self.punishment_coef,
            "shaped_reward_coef": self.shaped_reward_coef,
        }
        return env_info

    def _is_done(self):
        return self.mdp_state.timestep >= self.horizon or self.mdp.is_terminal(
            self.mdp_state
        )
