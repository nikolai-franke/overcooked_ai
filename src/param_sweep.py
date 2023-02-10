import wandb
from run import main

sweep_configuration = {
    "method": "bayes",
    "name": "ppo_reward_to_punishment_ratio",
    "metric": {
        "goal": "maximize",
        "name": "rollout/ep_sparse_r",
    },
    "parameters": {
        # "learning_rate": {"values": [1e-4]},
        # "net_n_neurons": {"values": [256]},
        # "net_n_layers": {"values": [2]},
        # "features_dim": {"values": [64]},
        # "batch_size": {"values": [1024]},
        "shaped_reward": {"min": 0.0, "max": 10.0},
        "shaped_punishment": {"min": -10.0, "max": 0.0},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="overcooked", entity="nikolai-franke"
    )

    wandb.agent(sweep_id, function=main, count=10)
