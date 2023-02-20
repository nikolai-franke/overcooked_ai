from stable_baselines3.common.callbacks import BaseCallback


class PunishmentCallback(BaseCallback):
    def __init__(
        self,
        start_timestep: int = 0,
        increase_duration: int = 7_000_000,
        verbose: int = 0,
    ):
        """
        Callback which sets the punishment coefficient of the environment at the start of each rollout.
        This coefficient linearly increases from 0.0 to 1.0 in the interval [start_timestep, start_timestep + increase_duration]

        :param start_timestep: the timestep at which to start increasing the coefficient
        :param increase_duration: the number of timesteps it takes to reach 1.0
        :param verbose: Verbosity level (see BaseCallback class)
        """
        super().__init__(verbose)
        self.start_timestep = start_timestep
        self.increase_duration = increase_duration

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if self.num_timesteps < self.start_timestep:
            self.training_env.set_attr("punishment_coef", 0.0)
        else:
            relative_timestep = self.num_timesteps - self.start_timestep
            value = min(relative_timestep / self.increase_duration, 1.0)
            self.training_env.set_attr("punishment_coef", value)
