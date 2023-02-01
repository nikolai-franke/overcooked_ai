import wandb
from run import main

sweep_configuration = {
    "method": "grid",
    "name": "base_param_sweep_2",
    "metric": {
        "goal": "maximize",
        "name": "rollout/ep_sparse_r",
    },
    "parameters": {
        "learning_rate": {"values": [1e-4, 5e-5]},
        "net_n_neurons": {"values": [128, 256]},
        "net_n_layers": {"values": [2, 3]},
        "features_dim": {"values": [128, 256]},
        # "batch_size": {"values": [1024, 2048]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="overcooked", entity="nikolai-franke"
    )

    wandb.agent(sweep_id, function=main)
