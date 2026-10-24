"""
Neural Network Architectures for Option Pricing & Error Correction.
Includes:
1. SimpleOptionANN: Dense -> ReLU -> Dense -> ReLU -> Output (as specified by Member 3).
2. ResidualOptionANN: Dedicated network learning the Black-Scholes error E = C_market - C_bs.
3. OptionLSTM: Sequence-aware LSTM combining historical price dynamics with contract features.
4. HybridShvimerSubModel1: Exact architecture from Shvimer & Zhu (2024) research paper.
"""

import torch
import torch.nn as nn


class SimpleOptionANN(nn.Module):
    """
    Simple Feedforward Artificial Neural Network (ANN) as requested:
    
    Input features (e.g., S, K, T, r, sigma, or S/K, T, r, sigma)
          |
        Dense (hidden_dim1, default 64)
          |
        ReLU
          |
        Dense (hidden_dim2, default 32)
          |
        ReLU
          |
        Output Dense (1 unit)
    
    Can be trained to predict:
    1. Direct Market Price: X -> C_market
    2. Residual Error:      X -> E
    """
    def __init__(self, input_dim: int = 5, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualOptionANN(nn.Module):
    """
    Residual Option Pricing Network:
    Directly models the pricing discrepancy of Black-Scholes:
        E = C_market - C_bs
        E_hat = f_theta(X)
        C_pred = C_bs + E_hat
    """
    def __init__(self, input_dim: int = 5, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super().__init__()
        self.error_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts the pricing error E_hat."""
        return self.error_net(x)

    def predict_price(self, x: torch.Tensor, c_bs: torch.Tensor) -> torch.Tensor:
        """Computes C_pred = C_bs + E_hat, enforcing non-negativity."""
        e_hat = self.forward(x)
        if c_bs.ndim == 1:
            c_bs = c_bs.unsqueeze(-1)
        if c_bs.shape != e_hat.shape:
            raise ValueError("c_bs must contain one baseline price per input row")
        c_pred = c_bs + e_hat
        return torch.clamp(c_pred, min=0.0)


class OptionLSTM(nn.Module):
    """
    Sequence-Aware LSTM for Option Pricing.
    Processes rolling temporal window of underlying stock movements (seq_len, seq_dim)
    and merges temporal representations with static contract parameters (static_dim).
    """
    def __init__(
        self,
        seq_input_dim: int = 2,
        static_input_dim: int = 5,
        lstm_hidden: int = 64,
        dense_hidden: int = 32,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        # LSTM processes past price & return trajectories
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout_prob
        )
        
        # Merge LSTM output with static contract features [S/K, T, r, sigma_hist, C_bs/K]
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + static_input_dim, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(dense_hidden, 1)
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        # x_seq: (batch_size, seq_len, seq_input_dim)
        lstm_out, (h_n, _) = self.lstm(x_seq)
        # Take the final hidden state of the top LSTM layer
        last_hidden = h_n[-1]  # (batch_size, lstm_hidden)
        
        # Concatenate temporal embedding with static contract terms
        merged = torch.cat([last_hidden, x_static], dim=-1)
        output = self.head(merged)
        return output


class HybridShvimerSubModel1(nn.Module):
    """
    Exact Topology of Sub-model 1 from Shvimer & Zhu (2024):
    - Input: [S/K, r, sigma, T, C_bs/K]
    - Layer 1: Dense (40 units) + Sigmoid
    - Dropout: 10%
    - Layer 2: Dense (40 units) + Sigmoid
    - Dropout: 10%
    - Output: Dense (1 unit) + Linear activation
    - Scaled target: C_market / K
    """
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.Sigmoid(),
            nn.Dropout(0.10),
            nn.Linear(40, 40),
            nn.Sigmoid(),
            nn.Dropout(0.10),
            nn.Linear(40, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
