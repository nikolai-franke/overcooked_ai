from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        env_info = self.locals["infos"][0]
        if "episode" in env_info.keys():
            episode_info = env_info["episode"]
            useless_actions = episode_info["ep_useless_a"]
            ep_shaped_r = episode_info["ep_shaped_r"]
            ep_sparse_r = episode_info["ep_sparse_r"]
            self.logger.record_mean("rollout/ep_useless_a", useless_actions)  # type: ignore
            self.logger.record_mean("rollout/ep_shaped_r", ep_shaped_r)  # type: ignore
            self.logger.record_mean("rollout/ep_sparse_r", ep_sparse_r)  # type: ignore
        return True
