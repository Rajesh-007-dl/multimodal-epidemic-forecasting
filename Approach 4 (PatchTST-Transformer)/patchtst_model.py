"""
patchtst_model.py
==================
PatchTST: Patch-based Temporal Transformer for Multi-Variate COVID-19 Forecasting.

Architecture based on:
  "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
  (Nie et al., ICLR 2023)

Key Design Choices:
  1. PATCHING: The input time series is divided into non-overlapping or overlapping
     sub-series patches (e.g., 16 days per patch). Each patch becomes one "token"
     in the attention sequence — giving the model local temporal context instead
     of raw point-level noise.

  2. CHANNEL INDEPENDENCE: Each input feature/channel is processed independently
     through the same transformer encoder. This prevents cross-channel overfitting
     on noisy multivariate data.

  3. DIRECT MULTI-HORIZON OUTPUT: A single linear head maps the CLS token output
     directly to H future timesteps — NO recursive step-by-step forecasting.
     This eliminates error accumulation completely.

Input:  (batch, seq_len, n_channels) — e.g., 14-day sequence with 10 features
Output: (batch, forecast_horizon, n_targets) — e.g., 15 future days for 2 targets
"""

import math
import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# 1. Positional Encoding
# =============================================================================
class FixedPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for patch sequence."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                   # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, num_patches, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# 2. Transformer Encoder Block
# =============================================================================
class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder layer: Multi-Head Attention + FFN + Residual."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn     = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        self.ffn      = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, attn_mask=None):
        # Multi-head self-attention with residual
        attn_out, attn_weights = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward with residual
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights


# =============================================================================
# 3. PatchTST Core Model
# =============================================================================
class PatchTST(nn.Module):
    """
    Full PatchTST model for multi-variate epidemic forecasting.

    Args:
        n_channels:       Number of input feature channels (e.g., 10)
        seq_len:          Input sequence length in days (e.g., 56)
        patch_size:       Number of time steps per patch token (e.g., 8)
        patch_stride:     Stride between patches (e.g., 4 for overlapping)
        d_model:          Transformer embedding dimension
        n_heads:          Number of attention heads
        n_layers:         Number of Transformer encoder layers
        d_ff:             Feed-forward hidden dimension
        dropout:          Dropout rate
        forecast_horizon: Number of future time steps to predict
        n_targets:        Number of target variables to forecast
    """
    def __init__(
        self,
        n_channels: int = 10,
        seq_len: int = 56,
        patch_size: int = 8,
        patch_stride: int = 4,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        forecast_horizon: int = 15,
        n_targets: int = 2
    ):
        super().__init__()
        self.n_channels      = n_channels
        self.seq_len         = seq_len
        self.patch_size      = patch_size
        self.patch_stride    = patch_stride
        self.d_model         = d_model
        self.forecast_horizon = forecast_horizon
        self.n_targets       = n_targets

        # Number of patches per channel
        self.num_patches = max(1, (seq_len - patch_size) // patch_stride + 1)

        # --- 1. Patch Projection: linearly embed each patch into d_model ---
        # Channel-independent: shared projection weights across all channels
        self.patch_proj = nn.Linear(patch_size, d_model)

        # --- 2. Learnable CLS token for sequence-level prediction ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- 3. Positional Encoding ---
        self.pos_enc = FixedPositionalEncoding(d_model, max_len=self.num_patches + 1, dropout=dropout)

        # --- 4. Transformer Encoder Stack ---
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # --- 5. Forecast Head: CLS token → H future steps for each target ---
        # Channel aggregation: combine all n_channels CLS representations
        self.channel_mix = nn.Linear(d_model * n_channels, d_model)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, forecast_horizon * n_targets)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor [batch, seq_len, n_channels]
            return_attention: If True, return attention weights for visualization
        Returns:
            forecasts: [batch, forecast_horizon, n_targets]
            (optionally) attn_weights
        """
        B, T, C = x.shape
        assert C == self.n_channels, f"Expected {self.n_channels} channels, got {C}"

        all_attn = []
        channel_cls_list = []

        # Process each channel independently
        for c in range(C):
            # Extract single channel: [B, T]
            x_c = x[:, :, c]

            # --- Patch Extraction ---
            patches = []
            for i in range(self.num_patches):
                start = i * self.patch_stride
                end   = start + self.patch_size
                if end > T:
                    # Pad last patch if needed
                    patch = torch.zeros(B, self.patch_size, device=x.device)
                    avail = T - start
                    if avail > 0:
                        patch[:, :avail] = x_c[:, start:T]
                else:
                    patch = x_c[:, start:end]
                patches.append(patch)

            # Stack to [B, num_patches, patch_size]
            patches = torch.stack(patches, dim=1)

            # --- Patch Projection → [B, num_patches, d_model] ---
            patch_emb = self.patch_proj(patches)

            # --- Prepend CLS token ---
            cls = self.cls_token.expand(B, -1, -1)       # [B, 1, d_model]
            patch_emb = torch.cat([cls, patch_emb], dim=1)  # [B, 1+num_patches, d_model]

            # --- Positional Encoding ---
            patch_emb = self.pos_enc(patch_emb)

            # --- Transformer Encoder ---
            attn_weights = None
            h = patch_emb
            for layer in self.encoder_layers:
                h, attn_weights = layer(h)
            h = self.norm(h)

            # Extract CLS token output: [B, d_model]
            cls_out = h[:, 0, :]
            channel_cls_list.append(cls_out)

            if return_attention and attn_weights is not None:
                all_attn.append(attn_weights)

        # --- Aggregate across channels: [B, C * d_model] → [B, d_model] ---
        channel_concat = torch.cat(channel_cls_list, dim=-1)   # [B, C * d_model]
        agg = self.channel_mix(channel_concat)                  # [B, d_model]

        # --- Forecast Head → [B, H * n_targets] → [B, H, n_targets] ---
        raw = self.forecast_head(agg)
        forecasts = raw.view(B, self.forecast_horizon, self.n_targets)

        if return_attention:
            return forecasts, all_attn
        return forecasts


# =============================================================================
# 4. Dataset for PatchTST (sliding window, multi-horizon targets)
# =============================================================================
class PatchTSTDataset(torch.utils.data.Dataset):
    """
    Sliding window dataset that produces:
      - x: (seq_len, n_channels)  — multi-modal input window
      - y: (forecast_horizon, n_targets) — future target values
    """
    def __init__(self, df, feature_cols, target_cols,
                 seq_len=56, forecast_horizon=15, scale=True):
        self.seq_len          = seq_len
        self.forecast_horizon = forecast_horizon
        self.scale            = scale

        features = df[feature_cols].values.astype(np.float32)
        targets  = df[target_cols].values.astype(np.float32)

        if scale:
            self.feat_mean = features.mean(axis=0)
            self.feat_std  = features.std(axis=0) + 1e-8
            self.tgt_scale = targets.max(axis=0) + 1e-8
            features = (features - self.feat_mean) / self.feat_std
            targets  = targets / self.tgt_scale
        else:
            self.tgt_scale = np.ones(len(target_cols))

        self.features = features
        self.targets  = targets

    def __len__(self):
        return len(self.features) - self.seq_len - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.forecast_horizon]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


# =============================================================================
# 5. Quick Sanity Check
# =============================================================================
if __name__ == "__main__":
    print("PatchTST module — sanity check...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        import torch
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    B, T, C = 8, 56, 10
    H = 15

    model = PatchTST(
        n_channels=C, seq_len=T, patch_size=8, patch_stride=4,
        d_model=128, n_heads=8, n_layers=3, d_ff=256,
        dropout=0.2, forecast_horizon=H, n_targets=2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")
    print(f"  Number of patches per channel: {model.num_patches}")

    x_dummy = torch.randn(B, T, C).to(device)
    out = model(x_dummy)
    print(f"  Output shape (forecasts): {out.shape}")  # [B, H, 2]
    assert out.shape == (B, H, 2), "Shape mismatch!"
    print("PatchTST module OK!")
