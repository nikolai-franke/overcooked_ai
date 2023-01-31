import os

import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
from cnn import CNN
from monitor import OvercookedMonitor
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from tensorboard_callback import TensorboardCallback

MODELS_DIR = "/home/babo/repos/overcooked_ai/models/"
RUNS_DIR = "/home/babo/repos/overcooked_ai/runs/"


def make_env(env_id, layout_name, horizon, rank, seed=None):
    def _init():
        mdp = OvercookedGridworld.from_layout_name(layout_name)
        base_env = OvercookedEnv.from_mdp(
            mdp,
            # start_state_fn=mdp.get_random_start_state_fn(True, 0.1),
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
        env = OvercookedMonitor(
            env=env,
            # info_keywords=(
            #     "combined_shaped_r",
            #     "combined_sparse_r",
            #     "useless_actions_by_agent",
            # ),
        )
        return env

    if seed is not None:
        set_random_seed(seed)
    return _init


if __name__ == "__main__":
    wandb_config = {
        "total_timesteps": 5_000_000,
    }
    run = wandb.init(
        project="overcooked",
        entity="nikolai-franke",
        config=wandb_config,
        sync_tensorboard=True,
    )
    num_cpu = 24
    env_id = "Overcooked-v0"
    layout_name = "asymmetric_advantages"
    horizon = 500
    save_dir = os.path.join(MODELS_DIR, "ppo")
    tensorboard_dir = os.path.join(RUNS_DIR, "ppo")

    env = SubprocVecEnv(
        [make_env(env_id, layout_name, horizon, i) for i in range(num_cpu)]
    )

    policy_kwargs = dict(
        features_extractor_class=CNN,
        features_extractor_kwargs=dict(features_dim=64),
        activation_fn=torch.nn.ReLU,
        normalize_images=False,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        "CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        verbose=2,
        batch_size=1024,
        tensorboard_log=tensorboard_dir,
    )

    obs = env.reset()
    model.learn(
        total_timesteps=wandb_config["total_timesteps"],
        callback=[
            WandbCallback(
                gradient_save_freq=0,
                model_save_path=os.path.join(save_dir, f"{run.id}"),
                verbose=2,
            ),
            TensorboardCallback(),
        ],
    )
    run.finish()  # type: ignore
    model.save(save_dir)

    test_env = make_env(env_id, layout_name, horizon, 0)()

    trajs = test_env.rollout(model, num_games=2)

    StateVisualizer().display_rendered_trajectory(
        trajs,
        img_directory_path="/home/babo/trajs/",
        ipython_display=False,
    )
