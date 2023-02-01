import os

import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
from cnn import CNN
from monitor import OvercookedMonitor
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from tensorboard_callback import TensorboardCallback

MODELS_DIR = "/home/babo/repos/overcooked_ai/models/"
RUNS_DIR = "/home/babo/repos/overcooked_ai/runs/"


def make_env(rank, env_id, layout_name, horizon, rew_shaping_params=None, seed=None):
    def _init():
        mdp = OvercookedGridworld.from_layout_name(
            layout_name, rew_shaping_params=rew_shaping_params
        )
        base_env = OvercookedEnv.from_mdp(
            mdp,
            horizon=horizon,
            info_level=0,
        )
        env = gym.make(
            env_id,
            base_env=base_env,
            featurize_fn=base_env.lossless_state_encoding_mdp,
        )
        if seed is not None:
            env.seed(seed + rank)
        env = OvercookedMonitor(env=env)
        return env

    if seed is not None:
        set_random_seed(seed)

    return _init


config_defaults = {
    "total_timesteps": 7_000_000,
    "batch_size": 1024,
    "learning_rate": 1e-4,
    "horizon": 500,
    "num_cpu": 24,
    "features_dim": 64,
    "net_n_neurons": 128,
    "net_n_layers": 2,
    "rew_placement_in_pot": 0.1,
    "rew_dish_pickup": 0.1,
    "rew_soup_pickup": 0.1,
    "rew_soup_cook": 0,
    "rew_useless_action": 0,
    "rew_wrong_delivery": 0,
    "layout_name": "simple_o",
}


def main():
    with wandb.init(config=config_defaults, sync_tensorboard=True) as run:  # type: ignore
        config = wandb.config
        env_id = "Overcooked-v0"

        save_dir = os.path.join(MODELS_DIR, "ppo")
        tensorboard_dir = os.path.join(RUNS_DIR, "ppo")

        total_timesteps = config.total_timesteps
        learning_rate = config.learning_rate
        batch_size = config.batch_size
        horizon = config.horizon
        num_cpu = config.num_cpu
        features_dim = config.features_dim
        net_n_layers = config.net_n_layers
        net_n_neurons = config.net_n_neurons

        layout_name = config.layout_name
        model = "ppo"

        # Reward shaping params
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": config.rew_placement_in_pot,
            "DISH_PICKUP_REWARD": config.rew_dish_pickup,
            "SOUP_PICKUP_REWARD": config.rew_soup_pickup,
            "SOUP_COOK_REWARD": config.rew_soup_cook,
            "USELESS_ACTION_REW": config.rew_useless_action,
            "WRONG_DELIVERY_REW": config.rew_wrong_delivery,
        }

        env = SubprocVecEnv(
            [
                make_env(
                    i,
                    env_id,
                    layout_name,
                    horizon,
                    rew_shaping_params,
                )
                for i in range(num_cpu)
            ]
        )

        net_structure = [net_n_neurons] * net_n_layers
        net_arch = dict(pi=net_structure, vf=net_structure)

        policy_kwargs = dict(
            features_extractor_class=CNN,
            features_extractor_kwargs=dict(features_dim=features_dim),
            activation_fn=torch.nn.ReLU,
            normalize_images=False,
            net_arch=net_arch,
        )

        if model == "ppo":
            model = PPO(
                "CnnPolicy",
                env=env,
                policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,
                verbose=2,
                batch_size=batch_size,
                tensorboard_log=tensorboard_dir,
            )
        else:
            raise NotImplementedError

        model.learn(
            total_timesteps=total_timesteps,
            callback=[
                TensorboardCallback(),
                WandbCallback(verbose=2),
            ],
        )
        model.save(save_dir)
