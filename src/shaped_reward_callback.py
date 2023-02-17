from stable_baselines3.common.callbacks import BaseCallback


class ShapedRewardCallback(BaseCallback):
    def __init__(self, max_timesteps: int = 2_000_000, verbose=0):
        super().__init__(verbose)
        self.max_timesteps = max_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        value = max(1 - (self.num_timesteps / self.max_timesteps), 0.0)
        self.training_env.set_attr("shaped_reward_coef", value)
