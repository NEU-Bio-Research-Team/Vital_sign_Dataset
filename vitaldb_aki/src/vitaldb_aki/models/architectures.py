"""Model architectures for AKI prediction."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class AKILSTM_Logits(nn.Module):
    """LSTM model for AKI prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F].
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        out, _ = self.lstm(x)  # [B,T,H]
        B, T, _ = out.shape
        idx = (lengths - 1).clamp(min=0)
        last = out[torch.arange(B, device=out.device), idx]
        last = self.dropout(last)
        return self.fc(last)


class BiLSTM_Logits(nn.Module):
    """Bidirectional LSTM model for AKI prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F].
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        out, _ = self.lstm(x)  # [B,T,2H]
        B, T, _ = out.shape
        idx = (lengths - 1).clamp(min=0)
        last = out[torch.arange(B, device=out.device), idx]
        last = self.dropout(last)
        return self.fc(last)


class AKIGRU_Logits(nn.Module):
    """Bidirectional GRU model for AKI prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F].
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        out, _ = self.gru(x)  # [B,T,2H]
        B, T, _ = out.shape
        t = torch.arange(T, device=out.device).unsqueeze(0).expand(B, T)
        m = (t < lengths.unsqueeze(1)).unsqueeze(-1).float()
        out = out * m
        pooled = out.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).float()
        return self.head(pooled)


class CausalConv1d(nn.Module):
    """Causal 1D convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        y = self.conv(x)
        if self.pad > 0:
            y = y[:, :, : -self.pad]
        return y


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""

    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(ch, ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + self.net(x)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        levels: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden, kernel_size=1)
        blocks = []
        for i in range(levels):
            blocks.append(TCNBlock(hidden, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F].
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        # x: [B,T,F] -> [B,F,T]
        x = x.transpose(1, 2)
        h = self.in_proj(x)
        h = self.tcn(h)  # [B,H,T]
        h = h.transpose(1, 2)  # [B,T,H]
        B, T, H = h.shape
        t = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)
        m = (t < lengths.unsqueeze(1)).unsqueeze(-1).float()
        h = h * m
        pooled = h.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).float()
        return self.head(pooled)


def build_model(model_name: str, input_dim: int, **kwargs) -> nn.Module:
    """Factory function to create models by name.

    Args:
        model_name: Model name ("lstm", "bilstm", "gru", "tcn").
        input_dim: Input dimension (number of features).
        **kwargs: Additional model-specific arguments.

    Returns:
        Model instance.

    Raises:
        ValueError: If model_name is unknown.
    """
    model_name = model_name.lower().strip()
    
    # Map parameters for TCN (hidden_dim -> hidden, num_layers -> levels)
    if model_name == "tcn":
        tcn_kwargs = kwargs.copy()
        if "hidden_dim" in tcn_kwargs:
            tcn_kwargs["hidden"] = tcn_kwargs.pop("hidden_dim")
        if "num_layers" in tcn_kwargs:
            tcn_kwargs["levels"] = tcn_kwargs.pop("num_layers")
        return TCNClassifier(input_dim, **tcn_kwargs)
    elif model_name == "lstm":
        return AKILSTM_Logits(input_dim, **kwargs)
    elif model_name == "bilstm":
        return BiLSTM_Logits(input_dim, **kwargs)
    elif model_name == "gru":
        return AKIGRU_Logits(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model_name={model_name}. Use one of: lstm, bilstm, gru, tcn")

