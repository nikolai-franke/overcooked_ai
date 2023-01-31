import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 0) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # type: ignore

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 25, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(25, 25, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(25, 25, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
