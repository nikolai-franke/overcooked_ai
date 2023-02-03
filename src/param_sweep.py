import wandb
from run import main

sweep_configuration = {
    "method": "grid",
    "name": "ppo_async_adv_rew",
    "metric": {
        "goal": "maximize",
        "name": "rollout/ep_sparse_r",
    },
    "parameters": {
        "learning_rate": {"values": [1e-4]},
        "net_n_neurons": {"values": [256]},
        "net_n_layers": {"values": [2]},
        "features_dim": {"values": [64]},
        "batch_size": {"values": [1024]},
        # "use_sde": {"values": [False, True]},
        "rew_placement_in_pot": {"values": [1, 5]},
        "rew_useless_action": {"values": [0, -0.1, -1]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="overcooked", entity="nikolai-franke"
    )

    wandb.agent(sweep_id, function=main)
