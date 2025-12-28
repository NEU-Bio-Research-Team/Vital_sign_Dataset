"""Model architectures for AKI prediction."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TCNAttentionClassifier(nn.Module):
    """TCN + Attention hybrid classifier.
    
    This model combines the strengths of TCN and attention mechanisms:
    - TCN blocks extract local temporal patterns efficiently using dilated convolutions
    - Temporal attention focuses on important time steps
    - Attention pooling provides weighted aggregation for final prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        """Initialize TCN + Attention classifier.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden state dimension.
            num_layers: Number of TCN blocks.
            kernel_size: Convolution kernel size for TCN blocks.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection: [B, T, F] -> [B, F, T] -> [B, H, T]
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # TCN stage: Extract local temporal patterns
        tcn_blocks = []
        for i in range(num_layers):
            dilation = 2 ** i
            tcn_blocks.append(TCNBlock(hidden_dim, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*tcn_blocks)

        # Temporal attention
        self.attention = TemporalAttention(hidden_dim, num_heads, dropout)

        # Attention pooling
        self.pool_proj = nn.Linear(hidden_dim, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F].
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        B, T, num_features = x.shape

        # Stage 1: TCN processing
        # Project input: [B, T, F] -> [B, F, T] -> [B, H, T]
        x_conv = x.transpose(1, 2)
        h = self.in_proj(x_conv)  # [B, H, T]
        h = self.tcn(h)  # [B, H, T]
        # Transpose back: [B, H, T] -> [B, T, H]
        h = h.transpose(1, 2)  # [B, T, H]

        # Stage 2: Temporal attention
        # Create mask for valid positions
        t = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)
        mask = (t < lengths.unsqueeze(1)).float()  # [B, T]
        
        # Apply temporal attention
        h = self.attention(h, mask)  # [B, T, H]

        # Stage 3: Attention pooling (learned weighted average)
        attn_scores = self.pool_proj(h)  # [B, T, 1]
        attn_scores = attn_scores.squeeze(-1)  # [B, T]

        # Mask invalid positions
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T]

        # Weighted sum
        pooled = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)  # [B, H]

        # Classification
        logits = self.classifier(pooled)  # [B, 1]

        return logits


class GatedDilatedBlock(nn.Module):
    """Gated dilated convolution block with WaveNet-style activation.
    
    Uses tanh + sigmoid gates for more expressive non-linearities.
    This is distinct from TCNBlock which uses ReLU activations.
    """

    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float):
        """Initialize gated dilated block.

        Args:
            ch: Number of channels.
            kernel_size: Convolution kernel size.
            dilation: Dilation rate.
            dropout: Dropout rate.
        """
        super().__init__()
        # Gated activation: split channels for tanh and sigmoid gates
        self.conv_filter = CausalConv1d(ch, ch, kernel_size, dilation)
        self.conv_gate = CausalConv1d(ch, ch, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        # 1x1 conv for residual connection
        self.residual = nn.Conv1d(ch, ch, kernel_size=1)
        # 1x1 conv for skip connection
        self.skip = nn.Conv1d(ch, ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated activation and residual connection.

        Args:
            x: Input tensor [B, C, T].

        Returns:
            Output tensor [B, C, T].
        """
        # Gated activation: filter * sigmoid(gate)
        filter_out = self.conv_filter(x)
        gate_out = self.conv_gate(x)
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        gated = self.dropout(gated)

        # Residual connection
        residual = self.residual(gated)
        out = x + residual

        return out


class DilatedConvClassifier(nn.Module):
    """Enhanced Dilated Convolution classifier with gated activations.
    
    This model is distinct from TCNClassifier:
    - Uses gated activation (tanh + sigmoid) instead of ReLU
    - Deeper architecture with more levels
    - Multi-scale feature extraction
    - Better pooling mechanism (last timestep + attention)
    
    Architecture inspired by WaveNet but adapted for classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 10,  # Deeper than TCN (which uses 6)
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_multiscale: bool = True,
    ):
        """Initialize enhanced dilated convolution classifier.

        Args:
            input_dim: Input dimension (signals + masks).
            hidden_dim: Hidden dimension size.
            num_layers: Number of dilated blocks (levels).
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
            use_multiscale: Whether to use multi-scale feature extraction.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_multiscale = use_multiscale

        # Input projection
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Multi-scale feature extraction (optional)
        if use_multiscale:
            # Parallel convolutions with different kernel sizes
            # Split hidden_dim into 3 parts (may not be exactly equal)
            part1 = hidden_dim // 3
            part2 = hidden_dim // 3
            part3 = hidden_dim - part1 - part2  # Remainder goes to part3
            self.multiscale_convs = nn.ModuleList([
                nn.Conv1d(hidden_dim, part1, kernel_size=3, padding=1),
                nn.Conv1d(hidden_dim, part2, kernel_size=5, padding=2),
                nn.Conv1d(hidden_dim, part3, kernel_size=7, padding=3),
            ])

        # Dilated convolution blocks with gated activation
        blocks = []
        for i in range(num_layers):
            dilation = 2 ** (i % 8)  # Cycle through dilations 1, 2, 4, 8, 16, 32, 64, 128
            blocks.append(GatedDilatedBlock(hidden_dim, kernel_size, dilation, dropout))
        self.dilated_blocks = nn.ModuleList(blocks)

        # Attention pooling (learned weighted average)
        self.attention_pool = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, 1, kernel_size=1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F] (signals + masks).
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        h = self.in_proj(x)  # [B, H, T]

        # Multi-scale feature extraction (optional)
        if self.use_multiscale:
            multiscale_features = []
            for conv in self.multiscale_convs:
                multiscale_features.append(F.relu(conv(h)))
            h = torch.cat(multiscale_features, dim=1)  # [B, H, T]

        # Apply dilated blocks
        for block in self.dilated_blocks:
            h = block(h)  # [B, H, T]

        # Attention pooling
        # h: [B, H, T] -> [B, T, H]
        h = h.transpose(1, 2)  # [B, T, H]

        # Create mask for valid positions
        B, T, H = h.shape
        t = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)
        mask = (t < lengths.unsqueeze(1)).float()  # [B, T]

        # Attention scores
        h_for_attn = h.transpose(1, 2)  # [B, H, T]
        attn_scores = self.attention_pool(h_for_attn).squeeze(1)  # [B, T]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T]

        # Weighted sum
        pooled = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)  # [B, H]

        # Classification
        logits = self.classifier(pooled)  # [B, 1]

        return logits


class DilatedRNNLayer(nn.Module):
    """Single dilated RNN layer that processes sequence with a specific dilation rate.
    
    The dilation is implemented by skipping time steps in the recurrent connections,
    allowing the model to capture long-range dependencies more efficiently.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dilation: int,
        cell_type: str = "lstm",
        dropout: float = 0.2,
    ):
        """Initialize dilated RNN layer.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden state dimension.
            dilation: Dilation rate (number of time steps to skip).
            cell_type: Type of RNN cell ("lstm" or "gru").
            dropout: Dropout rate.
        """
        super().__init__()
        self.dilation = dilation
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.lower()

        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
                dropout=0.0,  # No dropout at RNN level
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
                dropout=0.0,
            )
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}. Use 'lstm' or 'gru'")

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with dilation.

        Args:
            x: Input tensor [B, T, F].
            lengths: Sequence lengths [B].

        Returns:
            Output tensor [B, T, H].
        """
        B, T, num_features = x.shape

        # Apply dilation by subsampling the input sequence
        # For dilation d, we process every d-th time step
        if self.dilation > 1:
            # Create indices for dilated sampling
            dilated_indices = torch.arange(0, T, self.dilation, device=x.device)
            # Clamp to valid range
            dilated_indices = dilated_indices[dilated_indices < T]
            
            if len(dilated_indices) == 0:
                # If no valid indices, return zeros
                return torch.zeros(B, T, self.hidden_dim, device=x.device)
            
            # Subsample input
            x_dilated = x[:, dilated_indices, :]  # [B, T_dilated, num_features]
            
            # Adjust lengths for dilated sequence
            lengths_dilated = (lengths.float() / self.dilation).ceil().long().clamp(min=1)
            lengths_dilated = lengths_dilated.clamp(max=x_dilated.shape[1])
        else:
            x_dilated = x
            lengths_dilated = lengths

        # Process with RNN
        # Pack sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            x_dilated, lengths_dilated.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # [B, T_dilated, H]

        # Apply layer norm and dropout
        out = self.layer_norm(out)
        out = self.dropout(out)

        # Upsample back to original sequence length if dilated
        if self.dilation > 1:
            # Interpolate or pad to original length
            # Use nearest neighbor upsampling
            out = F.interpolate(
                out.transpose(1, 2),  # [B, H, T_dilated]
                size=T,
                mode="nearest",
            ).transpose(1, 2)  # [B, T, H]

        return out


class DilatedRNNClassifier(nn.Module):
    """Dilated RNN classifier with multi-scale temporal modeling.
    
    Uses multiple RNN layers with different dilation rates to capture
    dependencies at different time scales simultaneously.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dilations: list[int] | None = None,
        cell_type: str = "lstm",
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        """Initialize dilated RNN classifier.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden state dimension for each dilated layer.
            num_layers: Number of dilated RNN layers.
            dilations: List of dilation rates. If None, uses [1, 2, 4, 8, ...].
            cell_type: Type of RNN cell ("lstm" or "gru").
            dropout: Dropout rate.
            use_attention: Whether to use attention pooling (True) or mean pooling (False).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Default dilation rates: exponential growth
        if dilations is None:
            dilations = [2**i for i in range(num_layers)]
        else:
            assert len(dilations) == num_layers, f"len(dilations) must equal num_layers"
        self.dilations = dilations

        # Create dilated RNN layers
        self.dilated_layers = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.dilated_layers.append(
                DilatedRNNLayer(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    dilation=dilation,
                    cell_type=cell_type,
                    dropout=dropout,
                )
            )

        # Feature fusion: combine outputs from all dilated layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention pooling (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F].
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        B, T, num_features = x.shape

        # Process through all dilated layers in parallel
        # Each layer processes the same input with different dilation rates
        layer_outputs = []
        for i, layer in enumerate(self.dilated_layers):
            # First layer uses original input, subsequent layers use previous output
            # This creates a hierarchical multi-scale representation
            if i == 0:
                layer_input = x
            else:
                # Use previous layer output as input (allows hierarchical features)
                layer_input = layer_outputs[-1]
            
            out = layer(layer_input, lengths)  # [B, T, H]
            layer_outputs.append(out)

        # Concatenate outputs from all layers
        combined = torch.cat(layer_outputs, dim=-1)  # [B, T, H*num_layers]

        # Fuse features
        fused = self.fusion(combined)  # [B, T, H]

        # Pooling
        if self.use_attention:
            # Attention pooling
            attn_scores = self.attention(fused)  # [B, T, 1]
            # Mask out padding
            mask = (
                torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            attn_scores = attn_scores * mask + (1 - mask) * (-1e9)
            attn_weights = F.softmax(attn_scores, dim=1)
            pooled = torch.sum(fused * attn_weights, dim=1)  # [B, H]
        else:
            # Mean pooling with mask
            mask = (
                torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            masked = fused * mask
            pooled = masked.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).float()

        # Classification
        logits = self.classifier(pooled)  # [B, 1]

        return logits


class TemporalSynergyClassifier(nn.Module):
    """Temporal Synergy classifier combining TCN and Dilated RNN.
    
    This hybrid model combines the strengths of both architectures:
    - TCN blocks extract local temporal patterns efficiently using dilated convolutions
    - Dilated RNN layers model sequential dependencies from TCN features
    - Feature fusion combines multi-scale features from both stages
    - Attention pooling provides weighted aggregation for final prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        tcn_levels: int = 4,
        rnn_layers: int = 4,
        cell_type: str = "lstm",
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        """Initialize Temporal Synergy classifier.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden state dimension.
            tcn_levels: Number of TCN blocks.
            rnn_layers: Number of Dilated RNN layers.
            cell_type: Type of RNN cell ("lstm" or "gru").
            kernel_size: Convolution kernel size for TCN blocks.
            dropout: Dropout rate.
            use_attention: Whether to use attention pooling (True) or mean pooling (False).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tcn_levels = tcn_levels
        self.rnn_layers = rnn_layers
        self.use_attention = use_attention

        # Input projection: [B, T, F] -> [B, F, T] -> [B, H, T]
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # TCN stage: Extract local temporal patterns
        tcn_blocks = []
        for i in range(tcn_levels):
            dilation = 2 ** i
            tcn_blocks.append(TCNBlock(hidden_dim, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*tcn_blocks)

        # Dilated RNN stage: Model sequential dependencies
        self.rnn_layers_list = nn.ModuleList()
        for i in range(rnn_layers):
            dilation = 2 ** i
            self.rnn_layers_list.append(
                DilatedRNNLayer(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    dilation=dilation,
                    cell_type=cell_type,
                    dropout=dropout,
                )
            )

        # Feature fusion: Combine TCN output + all RNN layer outputs
        # TCN: [B, T, H], RNN layers: [B, T, H] × rnn_layers
        # Combined: [B, T, H × (1 + rnn_layers)]
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * (1 + rnn_layers), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention pooling (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F].
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        B, T, num_features = x.shape

        # Stage 1: TCN processing
        # Project input: [B, T, F] -> [B, F, T] -> [B, H, T]
        x_conv = x.transpose(1, 2)
        h_tcn = self.in_proj(x_conv)  # [B, H, T]
        h_tcn = self.tcn(h_tcn)  # [B, H, T]
        # Transpose back: [B, H, T] -> [B, T, H]
        h_tcn = h_tcn.transpose(1, 2)  # [B, T, H]

        # Stage 2: Dilated RNN processing
        # Process TCN output through dilated RNN layers
        rnn_outputs = []
        current_input = h_tcn

        for rnn_layer in self.rnn_layers_list:
            out = rnn_layer(current_input, lengths)  # [B, T, H]
            rnn_outputs.append(out)
            # Use previous layer output as input (hierarchical processing)
            current_input = out

        # Feature fusion: Concatenate TCN output + all RNN outputs
        # TCN: [B, T, H]
        # RNN: [B, T, H] × rnn_layers
        combined = [h_tcn] + rnn_outputs
        combined = torch.cat(combined, dim=-1)  # [B, T, H × (1 + rnn_layers)]

        # Fuse features
        fused = self.fusion(combined)  # [B, T, H]

        # Pooling
        if self.use_attention:
            # Attention pooling
            attn_scores = self.attention(fused)  # [B, T, 1]
            # Mask out padding
            mask = (
                torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            attn_scores = attn_scores * mask + (1 - mask) * (-1e9)
            attn_weights = F.softmax(attn_scores, dim=1)
            pooled = torch.sum(fused * attn_weights, dim=1)  # [B, H]
        else:
            # Mean pooling with mask
            mask = (
                torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            masked = fused * mask
            pooled = masked.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).float()

        # Classification
        logits = self.classifier(pooled)  # [B, 1]

        return logits


class WaveNetBlock(nn.Module):
    """WaveNet dilated convolution block with gated activation and skip connections.
    
    This is the core building block of WaveNet architecture. It differs from
    GatedDilatedBlock by explicitly returning both residual and skip outputs,
    allowing skip connections to be accumulated across the entire network.
    """

    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float):
        """Initialize WaveNet block.

        Args:
            ch: Number of channels.
            kernel_size: Convolution kernel size (typically 2 for WaveNet).
            dilation: Dilation rate.
            dropout: Dropout rate.
        """
        super().__init__()
        # Gated activation: split channels for tanh and sigmoid gates
        self.conv_filter = CausalConv1d(ch, ch, kernel_size, dilation)
        self.conv_gate = CausalConv1d(ch, ch, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        
        # 1x1 conv for residual connection (same channels)
        self.residual = nn.Conv1d(ch, ch, kernel_size=1)
        # 1x1 conv for skip connection (can have different output channels)
        self.skip = nn.Conv1d(ch, ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with gated activation, residual and skip connections.

        Args:
            x: Input tensor [B, C, T].

        Returns:
            Tuple of (residual_output, skip_output) both [B, C, T].
        """
        # Gated activation: filter * sigmoid(gate)
        filter_out = self.conv_filter(x)
        gate_out = self.conv_gate(x)
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        gated = self.dropout(gated)

        # Residual connection (for next layer)
        residual = self.residual(gated)
        residual_out = x + residual

        # Skip connection (accumulated across all layers)
        skip_out = self.skip(gated)

        return residual_out, skip_out


class WaveNetClassifier(nn.Module):
    """WaveNet-based classifier for AKI prediction.
    
    Implements the WaveNet architecture with:
    - Multiple stacks of dilated convolution layers
    - Gated activation units (tanh + sigmoid)
    - Residual connections within each block
    - Skip connections accumulated across all layers
    - Post-processing on skip connections for final prediction
    
    This is distinct from DilatedConvClassifier by:
    - Using skip connections as the primary output path
    - Stack structure with repeating dilation patterns
    - Post-processing layers on accumulated skip connections
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_stacks: int = 3,
        num_layers_per_stack: int = 10,
        kernel_size: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        """Initialize WaveNet classifier.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden state dimension.
            num_stacks: Number of WaveNet stacks (each stack repeats dilation pattern).
            num_layers_per_stack: Number of layers per stack.
            kernel_size: Convolution kernel size (typically 2 for WaveNet).
            dropout: Dropout rate.
            use_attention: Whether to use attention pooling on final features.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_stacks = num_stacks
        self.num_layers_per_stack = num_layers_per_stack
        self.use_attention = use_attention

        # Input projection
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Build WaveNet blocks with repeating dilation pattern
        # Each stack repeats: [1, 2, 4, 8, 16, 32, 64, 128, ...]
        self.blocks = nn.ModuleList()
        for stack in range(num_stacks):
            for layer in range(num_layers_per_stack):
                dilation = 2 ** (layer % 10)  # Repeat pattern every 10 layers
                self.blocks.append(
                    WaveNetBlock(hidden_dim, kernel_size, dilation, dropout)
                )

        # Post-processing on accumulated skip connections
        self.post_process = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # Attention pooling (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F].
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        B, T, num_features = x.shape

        # Project input: [B, T, F] -> [B, F, T] -> [B, H, T]
        x = x.transpose(1, 2)
        h = self.in_proj(x)  # [B, H, T]

        # Process through all WaveNet blocks
        # Accumulate skip connections
        skip_accum = None
        residual = h

        for block in self.blocks:
            residual, skip = block(residual)  # Both [B, H, T]
            if skip_accum is None:
                skip_accum = skip
            else:
                skip_accum = skip_accum + skip  # Accumulate skip connections

        # Post-process accumulated skip connections
        skip_accum = self.post_process(skip_accum)  # [B, H, T]

        # Transpose back: [B, H, T] -> [B, T, H]
        skip_accum = skip_accum.transpose(1, 2)  # [B, T, H]

        # Pooling
        if self.use_attention:
            # Attention pooling
            attn_scores = self.attention(skip_accum)  # [B, T, 1]
            # Mask out padding
            mask = (
                torch.arange(T, device=skip_accum.device).unsqueeze(0).expand(B, T)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            attn_scores = attn_scores * mask + (1 - mask) * (-1e9)
            attn_weights = F.softmax(attn_scores, dim=1)
            pooled = torch.sum(skip_accum * attn_weights, dim=1)  # [B, H]
        else:
            # Mean pooling with mask
            mask = (
                torch.arange(T, device=skip_accum.device).unsqueeze(0).expand(B, T)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            masked = skip_accum * mask
            pooled = masked.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).float()

        # Classification
        logits = self.classifier(pooled)  # [B, 1]

        return logits


class WaveNetRNNClassifier(nn.Module):
    """WaveNet + RNN hybrid classifier.
    
    This model combines the strengths of WaveNet and RNN architectures:
    - WaveNet blocks extract multi-scale temporal patterns using gated dilated convolutions
    - RNN layers model sequential dependencies from WaveNet features
    - Feature fusion combines multi-scale features from both stages
    - Attention pooling provides weighted aggregation for final prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_stacks: int = 3,
        num_layers_per_stack: int = 10,
        rnn_layers: int = 2,
        cell_type: str = "lstm",
        kernel_size: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        """Initialize WaveNet + RNN classifier.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden state dimension.
            num_stacks: Number of WaveNet stacks (each stack repeats dilation pattern).
            num_layers_per_stack: Number of layers per stack.
            rnn_layers: Number of RNN layers.
            cell_type: Type of RNN cell ("lstm" or "gru").
            kernel_size: Convolution kernel size (typically 2 for WaveNet).
            dropout: Dropout rate.
            use_attention: Whether to use attention pooling (True) or mean pooling (False).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_stacks = num_stacks
        self.num_layers_per_stack = num_layers_per_stack
        self.rnn_layers = rnn_layers
        self.use_attention = use_attention

        # Input projection
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Stage 1: WaveNet blocks with repeating dilation pattern
        # Each stack repeats: [1, 2, 4, 8, 16, 32, 64, 128, ...]
        self.wavenet_blocks = nn.ModuleList()
        for stack in range(num_stacks):
            for layer in range(num_layers_per_stack):
                dilation = 2 ** (layer % 10)  # Repeat pattern every 10 layers
                self.wavenet_blocks.append(
                    WaveNetBlock(hidden_dim, kernel_size, dilation, dropout)
                )

        # Post-processing on accumulated skip connections
        self.post_process = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # Stage 2: RNN layers for sequential modeling
        self.rnn = nn.ModuleList()
        for i in range(rnn_layers):
            if cell_type.lower() == "lstm":
                rnn_layer = nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                    dropout=0.0,
                )
            elif cell_type.lower() == "gru":
                rnn_layer = nn.GRU(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                    dropout=0.0,
                )
            else:
                raise ValueError(f"Unknown cell_type: {cell_type}. Use 'lstm' or 'gru'")
            self.rnn.append(rnn_layer)

        self.rnn_dropout = nn.Dropout(dropout)

        # Feature fusion: Combine WaveNet features + RNN features
        # WaveNet: [B, T, H], RNN: [B, T, H] × rnn_layers
        # Combined: [B, T, H × (1 + rnn_layers)]
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * (1 + rnn_layers), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention pooling (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F].
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        B, T, num_features = x.shape

        # Stage 1: WaveNet processing
        # Project input: [B, T, F] -> [B, F, T] -> [B, H, T]
        x_conv = x.transpose(1, 2)
        h = self.in_proj(x_conv)  # [B, H, T]

        # Process through all WaveNet blocks
        # Accumulate skip connections
        skip_accum = None
        residual = h

        for block in self.wavenet_blocks:
            residual, skip = block(residual)  # Both [B, H, T]
            if skip_accum is None:
                skip_accum = skip
            else:
                skip_accum = skip_accum + skip  # Accumulate skip connections

        # Post-process accumulated skip connections
        skip_accum = self.post_process(skip_accum)  # [B, H, T]

        # Transpose back: [B, H, T] -> [B, T, H]
        h_wavenet = skip_accum.transpose(1, 2)  # [B, T, H]

        # Stage 2: RNN processing
        # Process WaveNet output through RNN layers
        rnn_outputs = []
        current_input = h_wavenet

        for rnn_layer in self.rnn:
            out, _ = rnn_layer(current_input)  # [B, T, H]
            out = self.rnn_dropout(out)
            rnn_outputs.append(out)
            # Use previous layer output as input (hierarchical processing)
            current_input = out

        # Feature fusion: Concatenate WaveNet output + all RNN outputs
        # WaveNet: [B, T, H]
        # RNN: [B, T, H] × rnn_layers
        combined = [h_wavenet] + rnn_outputs
        combined = torch.cat(combined, dim=-1)  # [B, T, H × (1 + rnn_layers)]

        # Fuse features
        fused = self.fusion(combined)  # [B, T, H]

        # Pooling
        if self.use_attention:
            # Attention pooling
            attn_scores = self.attention(fused)  # [B, T, 1]
            # Mask out padding
            mask = (
                torch.arange(T, device=fused.device).unsqueeze(0).expand(B, T)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            attn_scores = attn_scores * mask + (1 - mask) * (-1e9)
            attn_weights = F.softmax(attn_scores, dim=1)
            pooled = torch.sum(fused * attn_weights, dim=1)  # [B, H]
        else:
            # Mean pooling with mask
            mask = (
                torch.arange(T, device=fused.device).unsqueeze(0).expand(B, T)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            masked = fused * mask
            pooled = masked.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).float()

        # Classification
        logits = self.classifier(pooled)  # [B, 1]

        return logits


class TemporalAttention(nn.Module):
    """Temporal self-attention mechanism with mask-aware processing."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        """Initialize temporal attention.

        Args:
            hidden_dim: Hidden dimension size.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with mask-aware attention.

        Args:
            x: Input tensor [B, T, H].
            mask: Binary mask [B, T] (1 = valid, 0 = missing).

        Returns:
            Attended features [B, T, H].
        """
        B, T, H = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        # Apply mask: set invalid positions to -inf
        attn_mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
        scores = scores.masked_fill(~attn_mask.bool(), float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, H)  # [B, T, H]
        out = self.out_proj(out)

        return out


class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal feature extraction using different kernel sizes."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        """Initialize multi-scale encoder.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        # Different kernel sizes for multi-scale features
        # Split hidden_dim into 3 parts (may not be exactly equal)
        part1 = hidden_dim // 3
        part2 = hidden_dim // 3
        part3 = hidden_dim - part1 - part2  # Remainder goes to part3
        
        self.conv1 = nn.Conv1d(input_dim, part1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, part2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, part3, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F].

        Returns:
            Multi-scale features [B, T, H].
        """
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)

        # Multi-scale convolutions
        f1 = F.relu(self.conv1(x))  # [B, H/3, T]
        f2 = F.relu(self.conv2(x))
        f3 = F.relu(self.conv3(x))

        # Concatenate
        out = torch.cat([f1, f2, f3], dim=1)  # [B, H, T]
        out = self.bn(out)
        out = self.dropout(out)

        # [B, H, T] -> [B, T, H]
        return out.transpose(1, 2)


class AttentionAKIPredictor(nn.Module):
    """Attention-based AKI prediction model.

    Architecture:
    1. Input projection (handles signals + masks)
    2. Multi-scale temporal encoder (optional)
    3. Bidirectional LSTM for temporal modeling
    4. Temporal attention to focus on important time steps
    5. Attention pooling
    6. Classification head with class imbalance handling
    """

    def __init__(
        self,
        input_dim: int,  # signals + masks (e.g., 10 for 5 signals)
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_multiscale: bool = True,
    ):
        """Initialize attention-based predictor.

        Args:
            input_dim: Input dimension (signals + masks).
            hidden_dim: Hidden dimension size.
            num_layers: Number of LSTM layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            use_multiscale: Whether to use multi-scale encoder.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-scale encoder (optional)
        if use_multiscale:
            self.multiscale = MultiScaleTemporalEncoder(hidden_dim, hidden_dim, dropout)
        else:
            self.multiscale = None

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Will be doubled by bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Temporal attention
        self.attention = TemporalAttention(hidden_dim, num_heads, dropout)

        # Attention pooling
        self.pool_proj = nn.Linear(hidden_dim, 1)

        # Classification head with deeper network
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F] (signals + masks).
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        B, T, num_features = x.shape

        # Extract mask from input (assuming last num_features//2 channels are masks)
        n_signals = num_features // 2
        signals = x[:, :, :n_signals]
        masks = x[:, :, n_signals:]  # [B, T, n_signals]

        # Create temporal mask (valid if any signal is present)
        temporal_mask = masks.sum(dim=2) > 0  # [B, T]

        # Project input
        h = self.input_proj(x)  # [B, T, H]

        # Multi-scale encoding (optional)
        if self.multiscale is not None:
            h = self.multiscale(h)

        # LSTM encoding
        h, _ = self.lstm(h)  # [B, T, H]

        # Temporal attention
        h = self.attention(h, temporal_mask)  # [B, T, H]

        # Attention pooling (learned weighted average)
        attn_scores = self.pool_proj(h)  # [B, T, 1]
        attn_scores = attn_scores.squeeze(-1)  # [B, T]

        # Mask invalid positions
        attn_scores = attn_scores.masked_fill(~temporal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T]

        # Weighted sum
        pooled = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)  # [B, H]

        # Classification
        logits = self.classifier(pooled)  # [B, 1]

        return logits


class TransformerAKIPredictor(nn.Module):
    """Transformer-based AKI prediction model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        max_len: int = 3600,  # 60 minutes at 1Hz
    ):
        """Initialize transformer-based predictor.

        Args:
            input_dim: Input dimension (signals + masks).
            hidden_dim: Hidden dimension size.
            num_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            max_len: Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding (add 1 to max_len to handle edge cases with padding)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len + 1, hidden_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, F] (signals + masks).
            lengths: Sequence lengths [B].

        Returns:
            Logits [B, 1].
        """
        B, T, num_features = x.shape

        # Project input
        h = self.input_proj(x)  # [B, T, H]

        # Add positional encoding (before CLS token)
        # Clamp T to max_len to avoid index out of bounds
        pos_len = min(T, self.pos_encoding.shape[1])
        h = h + self.pos_encoding[:, :pos_len, :]  # [B, T, H]

        # Add CLS token (with zero positional encoding, or use first position)
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, H]
        # CLS token doesn't need positional encoding, or use position 0
        h = torch.cat([cls, h], dim=1)  # [B, T+1, H]

        # Create attention mask (1 = valid, 0 = padding)
        # CLS token is always valid, then valid positions based on lengths
        max_len = int(lengths.max().item())
        mask = torch.zeros(B, T + 1, dtype=torch.bool, device=x.device)
        mask[:, 0] = True  # CLS token
        for i in range(B):
            valid_len = int(lengths[i].item())
            if valid_len > 0:
                mask[i, 1 : valid_len + 1] = True

        # Transformer encoding
        h = self.transformer(h, src_key_padding_mask=~mask)

        # Use CLS token for classification
        cls_output = h[:, 0, :]  # [B, H]

        # Classification
        logits = self.classifier(cls_output)  # [B, 1]

        return logits


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
    elif model_name == "attention":
        # Extract attention-specific parameters
        attention_kwargs = kwargs.copy()
        num_heads = attention_kwargs.pop("num_heads", 4)
        use_multiscale = attention_kwargs.pop("use_multiscale", True)
        return AttentionAKIPredictor(
            input_dim,
            num_heads=num_heads,
            use_multiscale=use_multiscale,
            **attention_kwargs,
        )
    elif model_name == "transformer":
        # Extract transformer-specific parameters
        transformer_kwargs = kwargs.copy()
        num_heads = transformer_kwargs.pop("num_heads", 8)
        max_len = transformer_kwargs.pop("max_len", 3600)
        return TransformerAKIPredictor(
            input_dim,
            num_heads=num_heads,
            max_len=max_len,
            **transformer_kwargs,
        )
    elif model_name == "dilated_conv":
        # Extract dilated_conv-specific parameters
        dilated_kwargs = kwargs.copy()
        use_multiscale = dilated_kwargs.pop("use_multiscale", True)
        # Map num_layers if provided
        if "num_layers" in dilated_kwargs:
            dilated_kwargs["num_layers"] = dilated_kwargs["num_layers"]
        return DilatedConvClassifier(
            input_dim,
            use_multiscale=use_multiscale,
            **dilated_kwargs,
        )
    elif model_name == "dilated_rnn":
        # Extract dilated_rnn-specific parameters
        dilated_rnn_kwargs = kwargs.copy()
        cell_type = dilated_rnn_kwargs.pop("cell_type", "lstm")
        use_attention = dilated_rnn_kwargs.pop("use_attention", True)
        dilations = dilated_rnn_kwargs.pop("dilations", None)
        # Default to 4 layers if not specified
        if "num_layers" not in dilated_rnn_kwargs:
            dilated_rnn_kwargs["num_layers"] = 4
        return DilatedRNNClassifier(
            input_dim,
            cell_type=cell_type,
            use_attention=use_attention,
            dilations=dilations,
            **dilated_rnn_kwargs,
        )
    elif model_name == "wavenet":
        # Extract wavenet-specific parameters
        wavenet_kwargs = kwargs.copy()
        use_attention = wavenet_kwargs.pop("use_attention", True)
        num_stacks = wavenet_kwargs.pop("num_stacks", 3)
        num_layers_per_stack = wavenet_kwargs.pop("num_layers_per_stack", 10)
        # Map num_layers to num_layers_per_stack if provided
        if "num_layers" in wavenet_kwargs:
            num_layers_per_stack = wavenet_kwargs.pop("num_layers")
        return WaveNetClassifier(
            input_dim,
            num_stacks=num_stacks,
            num_layers_per_stack=num_layers_per_stack,
            use_attention=use_attention,
            **wavenet_kwargs,
        )
    elif model_name == "temporal_synergy":
        # Extract temporal_synergy-specific parameters
        synergy_kwargs = kwargs.copy()
        # Remove num_layers if present (not used by temporal_synergy)
        synergy_kwargs.pop("num_layers", None)
        tcn_levels = synergy_kwargs.pop("tcn_levels", 4)
        rnn_layers = synergy_kwargs.pop("rnn_layers", 4)
        cell_type = synergy_kwargs.pop("cell_type", "lstm")
        use_attention = synergy_kwargs.pop("use_attention", True)
        return TemporalSynergyClassifier(
            input_dim,
            tcn_levels=tcn_levels,
            rnn_layers=rnn_layers,
            cell_type=cell_type,
            use_attention=use_attention,
            **synergy_kwargs,
        )
    elif model_name == "tcn_attention":
        # Extract tcn_attention-specific parameters
        tcn_attn_kwargs = kwargs.copy()
        num_heads = tcn_attn_kwargs.pop("num_heads", 4)
        # Map num_layers if provided (TCN uses num_layers, not levels)
        if "num_layers" not in tcn_attn_kwargs:
            tcn_attn_kwargs["num_layers"] = 6
        return TCNAttentionClassifier(
            input_dim,
            num_heads=num_heads,
            **tcn_attn_kwargs,
        )
    elif model_name == "wavenet_rnn":
        # Extract wavenet_rnn-specific parameters
        wavenet_rnn_kwargs = kwargs.copy()
        use_attention = wavenet_rnn_kwargs.pop("use_attention", True)
        num_stacks = wavenet_rnn_kwargs.pop("num_stacks", 3)
        num_layers_per_stack = wavenet_rnn_kwargs.pop("num_layers_per_stack", 10)
        rnn_layers = wavenet_rnn_kwargs.pop("rnn_layers", 2)
        cell_type = wavenet_rnn_kwargs.pop("cell_type", "lstm")
        # Map num_layers to num_layers_per_stack if provided
        if "num_layers" in wavenet_rnn_kwargs and "num_layers_per_stack" not in wavenet_rnn_kwargs:
            num_layers_per_stack = wavenet_rnn_kwargs.pop("num_layers")
        return WaveNetRNNClassifier(
            input_dim,
            num_stacks=num_stacks,
            num_layers_per_stack=num_layers_per_stack,
            rnn_layers=rnn_layers,
            cell_type=cell_type,
            use_attention=use_attention,
            **wavenet_rnn_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown model_name={model_name}. Use one of: lstm, bilstm, gru, tcn, attention, transformer, dilated_conv, dilated_rnn, wavenet, temporal_synergy, tcn_attention, wavenet_rnn"
        )

