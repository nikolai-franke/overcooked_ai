from stable_baselines3.common.callbacks import BaseCallback


class ShapedRewardCallback(BaseCallback):
    def __init__(self, duration: int = 2_000_000, verbose: int = 0):
        """
        Callback which decreases the shaped reward coefficient of the environment from 1.0 to 0.0.

        :param duration: Duration it takes to decrease the coefficient from 1.0 to 0.0
        :param verbose: Verbosity level (see BaseCallback class)
        """
        super().__init__(verbose)
        self.duration = duration

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        value = max(1 - (self.num_timesteps / self.duration), 0.0)
        self.training_env.set_attr("shaped_reward_coef", value)
