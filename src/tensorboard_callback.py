from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        env_info = self.locals["infos"]
        if "episode" in env_info[0].keys():
            for e in env_info:
                self.logger.record_mean("rollout/ep_useless_a", e["episode"]["ep_useless_a"])  # type: ignore
                self.logger.record_mean("rollout/ep_shaped_r", e["episode"]["ep_shaped_r"])  # type: ignore
                self.logger.record_mean("rollout/ep_sparse_r", e["episode"]["ep_sparse_r"])  # type: ignore
                self.logger.record_mean(
                    "rollout/ep_punishment", e["episode"]["ep_punishment"]
                )
                self.logger.record_mean("rollout/ep_wrong_d", e["episode"]["ep_wrong_d"])  # type: ignore
                self.logger.record_mean(
                    "rollout/ep_collisions", e["episode"]["ep_collisions"]
                )
        return True
