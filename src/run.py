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
from punishment_callback import PunishmentCallback
from shaped_reward_callback import ShapedRewardCallback
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
        set_random_seed(seed, using_cuda=True)

    return _init


config_defaults = {
    "seed": None,
    "total_timesteps": 7_000_000,
    "batch_size": 1920,
    "learning_rate": 6e-4,
    "lr_linear_schedule": True,
    "horizon": 400,
    "num_cpu": 24,
    "features_dim": 64,
    "net_n_neurons": 64,
    "net_n_layers": 2,
    "placement_in_pot_reward": 3,
    "dish_pickup_reward": 3,
    "soup_pickup_reward": 5,
    "soup_cook_reward": 0,
    "collision_reward": -3,
    "useless_action_reward": -0.1,
    "wrong_delivery_reward": -5,
    "layout_name": "cramped_room_small",
    "model_name": "ppo",
    "discount_factor": 0.98,
    "n_steps": 400,
    "n_epochs": 10,
    "ent_coef": 0.0,
    "clip_range": 0.2,
    "clip_range_schedule": True,
    "max_grad_norm": 0.5,
    "shaped_rewards_horizon": None,
    "punishment_start": None,
    "punishment_inv_horizon": None,
    "punishment_mode": None,
    "punishment_exponent": 1,
}


def main(config=config_defaults):
    with wandb.init(config=config, entity="nikolai-franke", project="overcooked", sync_tensorboard=True) as run:  # type: ignore
        config = wandb.config
        env_id = "Overcooked-v1"

        seed = config.seed
        total_timesteps = config.total_timesteps
        learning_rate = config.learning_rate
        lr_linear_schedule = config.lr_linear_schedule
        batch_size = config.batch_size
        horizon = config.horizon
        num_cpu = config.num_cpu
        features_dim = config.features_dim
        net_n_layers = config.net_n_layers
        net_n_neurons = config.net_n_neurons
        model_name = config.model_name
        n_epochs = config.n_epochs
        n_steps = config.n_steps
        ent_coef = config.ent_coef
        clip_range = config.clip_range
        clip_range_schedule = config.clip_range_schedule
        max_grad_norm = config.max_grad_norm
        shaped_rewards_horizon = config.shaped_rewards_horizon
        punishment_start = config.punishment_start
        punishment_inv_horizon = config.punishment_inv_horizon
        punishment_exponent = config.punishment_exponent
        discount_factor = config.discount_factor

        layout_name = config.layout_name
        punishment_mode = config.punishment_mode

        if lr_linear_schedule:
            initial_lr = learning_rate
            learning_rate = lambda progress_remaining: progress_remaining * initial_lr

        if clip_range_schedule:
            initial_clip_range = clip_range
            clip_range = (
                lambda progress_remaining: progress_remaining * initial_clip_range
            )

        if punishment_mode == "no_punishment":
            punishment_start = None
            punishment_inv_horizon = None
        elif punishment_mode == "full":
            punishment_start = 0
            punishment_inv_horizon = 1
        elif punishment_mode == "full_half":
            punishment_start = int(0.5 * total_timesteps)
            punishment_inv_horizon = 1
        elif punishment_mode == "slow":
            punishment_start = 0
            punishment_inv_horizon = total_timesteps
        elif punishment_mode == "slow_half":
            punishment_start = 0
            punishment_inv_horizon = int(0.5 * total_timesteps)

        # Reward shaping params
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": config.placement_in_pot_reward,
            "DISH_PICKUP_REWARD": config.dish_pickup_reward,
            "SOUP_PICKUP_REWARD": config.soup_pickup_reward,
            "SOUP_COOK_REWARD": config.soup_cook_reward,
            "COLLISION_REW": config.collision_reward,
            "USELESS_ACTION_REW": config.useless_action_reward,
            "WRONG_DELIVERY_REW": config.wrong_delivery_reward,
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
                    seed=seed,
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
            normalize_images=True,
            net_arch=net_arch,
        )

        if model_name == "ppo":
            model = PPO(
                "CnnPolicy",
                env=env,
                policy_kwargs=policy_kwargs,
                learning_rate=learning_rate,
                verbose=2,
                n_steps=n_steps,
                clip_range=clip_range,
                ent_coef=ent_coef,
                max_grad_norm=max_grad_norm,
                n_epochs=n_epochs,
                batch_size=batch_size,
                tensorboard_log=tensorboard_dir,
                gamma=discount_factor,
            )
        else:
            raise NotImplementedError

        callback_list = []
        if shaped_rewards_horizon is not None:
            callback_list.append(ShapedRewardCallback(shaped_rewards_horizon))

        if punishment_start is not None:
            assert punishment_inv_horizon is not None
            callback_list.append(
                PunishmentCallback(
                    punishment_start, punishment_inv_horizon, punishment_exponent
                )
            )

        callback_list.append(TensorboardCallback())
        callback_list.append(WandbCallback(verbose=2))

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
        )

        model.save(save_dir)

        # create test environment and record video of trained agent
        test_seed = seed + 1000 if seed is not None else None
        test_env = make_env(
            0, env_id, layout_name, horizon, rew_shaping_params, seed=test_seed
        )()
        model = PPO.load(save_dir, test_env)

        obs = model.env.reset()
        img = model.env.render(mode="rgb_array")
        done = False
        images = [img]
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = model.env.step(action)
            img = model.env.render("rgb_array")
            images.append(img)

        wandb.log({"video": wandb.Video(np.array(images), fps=10)})


if __name__ == "__main__":
    main()
