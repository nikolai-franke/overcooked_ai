from stable_baselines3.common.callbacks import BaseCallback


class PunishmentCallback(BaseCallback):
    def __init__(
        self,
        start_timestep: int = 0,
        increase_duration: int = 7_000_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.start_timestep = start_timestep
        self.increase_duration = increase_duration

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if self.num_timesteps < self.start_timestep:
            self.training_env.set_attr("punishment_coef", 0.0)
        elif self.num_timesteps <= self.increase_duration:
            relative_timestep = self.increase_duration - self.num_timesteps
            value = min(self.num_timesteps / relative_timestep, 1.0)
            self.training_env.set_attr("punishment_coef", value)
