import os

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
from cnn import CNN
from monitor import OvercookedMonitor
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from static import MODELS_DIR, RUNS_DIR
from tensorboard_callback import TensorboardCallback


def make_env(
    rank, env_id, layout_name, horizon=500, rew_shaping_params=None, seed=None
):
    def _init():
        mdp = OvercookedGridworld.from_layout_name(
            layout_name, rew_shaping_params=rew_shaping_params
        )
        env = gym.make(
            env_id,
            mdp=mdp,
            horizon=horizon,
        )
        if seed is not None:
            env.seed(seed + rank)
        env = OvercookedMonitor(env=env)
        return env

    if seed is not None:
        set_random_seed(seed)

    return _init


config_defaults = {
    "total_timesteps": 500_000,
    "batch_size": 1024,
    "learning_rate": 1e-4,
    "horizon": 500,
    "num_cpu": 24,
    "features_dim": 64,
    "net_n_neurons": 256,
    "net_n_layers": 2,
    "shaped_reward": 1,
    "shaped_punishment": 0,
    "layout_name": "asymmetric_advantages",
    "model_name": "ppo",
    "n_epochs": 10,
}


def main(config=config_defaults):
    with wandb.init(config=config, entity="nikolai-franke", project="overcooked", sync_tensorboard=True) as run:  # type: ignore
        config = wandb.config
        env_id = "Overcooked-v1"

        total_timesteps = config.total_timesteps
        learning_rate = config.learning_rate
        batch_size = config.batch_size
        horizon = config.horizon
        num_cpu = config.num_cpu
        features_dim = config.features_dim
        net_n_layers = config.net_n_layers
        net_n_neurons = config.net_n_neurons
        model_name = config.model_name
        layout_name = config.layout_name
        n_epochs = config.n_epochs

        # Reward shaping params
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": config.shaped_reward,
            "DISH_PICKUP_REWARD": config.shaped_reward,
            "SOUP_PICKUP_REWARD": config.shaped_reward,
            "SOUP_COOK_REWARD": config.shaped_reward,
            "USELESS_ACTION_REW": config.shaped_punishment,
            "WRONG_DELIVERY_REW": config.shaped_punishment,
        }

        model_save_name = f"{model_name}/{run.name}_{run.id}"
        save_dir = os.path.join(MODELS_DIR, model_save_name)
        tensorboard_dir = os.path.join(RUNS_DIR, model_save_name)

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

        if model_name == "ppo":
            model = PPO(
                "CnnPolicy",
                env=env,
                policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,
                verbose=2,
                n_epochs=n_epochs,
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

        # create test environment and record video of trained agent
        test_env = make_env(0, env_id, layout_name, horizon, rew_shaping_params)()
        model = PPO.load(save_dir, test_env)

        obs = model.env.reset()
        img = model.env.render(mode="rgb_array")
        done = False
        images = []
        while not done:
            images.append(img)
            action, _ = model.predict(obs)
            obs, _, done, _ = model.env.step(action)
            img = model.env.render("rgb_array")

        wandb.log({"video": wandb.Video(np.array(images), fps=10)})


if __name__ == "__main__":
    main()
