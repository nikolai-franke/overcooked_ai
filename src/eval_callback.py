from typing import Any, Dict

from stable_baselines3.common.callbacks import EvalCallback


class OvercookedEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _log_success_callback(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        info = locals_["info"]

        if "episode" in info[0].keys():
            for e in info:
                self.logger.record_mean("eval/ep_useless_a", e["episode"]["ep_useless_a"])  # type: ignore
                self.logger.record_mean("eval/ep_shaped_r", e["episode"]["ep_shaped_r"])  # type: ignore
                self.logger.record_mean("eval/ep_sparse_r", e["episode"]["ep_sparse_r"])  # type: ignore
                self.logger.record_mean(
                    "eval/ep_punishment", e["episode"]["ep_punishment"]
                )
                self.logger.record_mean("eval/ep_wrong_d", e["episode"]["ep_wrong_d"])  # type: ignore
                self.logger.record_mean(
                    "eval/ep_collisions", e["episode"]["ep_collisions"]
                )
