import wandb
from run import main

sweep_configuration = {
    "method": "grid",
    "name": "test_rewards_3_seeds_no_lr_schedule",
    "metric": {
        "goal": "maximize",
        "name": "rollout/ep_sparse_r",
    },
    "parameters": {
        # "shaped_reward": {"values": [3]},
        "seed": {"values": [0, 100, 200]},
        "ent_coef": {"values": [0.01]},
        "placement_in_pot_reward": {"values": [3, 5]},
        "dish_pickup_reward": {"values": [3, 5]},
        # "wrong_delivery_reward": {"values": [0.0, -0.1, -1.0]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="overcooked", entity="nikolai-franke"
    )

    wandb.agent(sweep_id, function=main)
