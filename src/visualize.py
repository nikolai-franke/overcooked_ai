import imageio as iio
import pygame
from pygifsicle import optimize
from stable_baselines3 import PPO, SAC

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


def save_trajectory_as_gif(trajectories, path: str, trajectory_idx: int = 0):
    visualizer = StateVisualizer()
    states = trajectories["ep_states"][trajectory_idx]
    grid = trajectories["mdp_params"][trajectory_idx]["terrain"]
    # hud_data = visualizer.default_hud_data_from_trajectories(trajectories)
    hud_data = visualizer.default_hud_data_sparse(trajectories)

    images = []
    for state, hud in zip(states, hud_data):
        surface = visualizer.render_state(state, grid, hud)
        image = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        images.append(image)

    iio.mimsave(path, images)
    optimize(path)


if __name__ == "__main__":
    from run import make_env

    model_path = (
        "/home/babo/repos/overcooked_ai/models/ppo/ethereal-sweep-1_oqn5gt7x.zip"
    )
    env_id = "Overcooked-v0"
    layout_name = "asymmetric_advantages"
    env = make_env(0, env_id, layout_name)()
    model = PPO.load(model_path, env=env)
    trajs = env.rollout(model)

    save_trajectory_as_gif(trajs, "./trajs.gif")
