# models/dml_model.py

import torch
import torch.nn as nn

class OptionMLP(nn.Module):
    """
    Multi-layer perceptron for European call option pricing.
    Designed to remain fully differentiable so we can extract Greeks via autograd.
    """
    def __init__(self, input_dim: int = 5, hidden_dims: tuple = (64, 64, 32)):
        """
        Args:
            input_dim: Number of input features (S, K, T, r, sigma).
            hidden_dims: Sizes of hidden layers.
        """
        super(OptionMLP, self).__init__()

        # Build sequential network: [Linear → SiLU] × N → Linear
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.SiLU())  # Smooth activation helps gradient-based losses
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))  # Output: scalar option price

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier init for weights, zero init for biases."""
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, 5), columns = [S, K, T, r, sigma].
        
        Returns:
            Tensor of shape (batch_size,) with predicted option prices.
        """
        price = self.model(x).squeeze(-1)
        return price
