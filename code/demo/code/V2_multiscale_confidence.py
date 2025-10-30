import torch
import torch.nn as nn
from typing import Dict

class V2MultiScaleConfidence(nn.Module):
    """
    V2 multiscale model without 1h GRU.
    Components:
      - gru_1d on daily window (input_size=1, hidden_size=H)
      - lstm_1w on weekly window (input_size=1, hidden_size=H)
      - transformer_1m on monthly window with input projection to H
      - feature fusion and regressor
    Forward expects dict with keys: '1d', '1w', '1m'.
    Shapes:
      - '1d': (B, 24, 1)
      - '1w': (B, 168, 1)
      - '1m': (B, Tm, 1) with Tm>=24*30
    Returns: (B, 1)
    """
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        H = hidden_size
        self.hidden_size = H

        self.gru_1d = nn.GRU(input_size=1, hidden_size=H, batch_first=True)
        self.lstm_1w = nn.LSTM(input_size=1, hidden_size=H, batch_first=True)

        self.input_projection = nn.Linear(1, H)
        self.transformer_1m = nn.Transformer(d_model=H, nhead=4, num_encoder_layers=2, batch_first=True)

        self.feature_fusion = nn.Linear(H * 3, H)
        self.fc = nn.Linear(H, 1)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_1d = x['1d']  # (B, 24, 1)
        x_1w = x['1w']  # (B, 168, 1)
        x_1m = x['1m']  # (B, Tm, 1)

        # Daily GRU -> final hidden
        _, h_1d = self.gru_1d(x_1d)
        h_1d = h_1d[-1]

        # Weekly LSTM -> final hidden
        _, (h_1w, _) = self.lstm_1w(x_1w)
        h_1w = h_1w[-1]

        # Monthly Transformer (encoder-only with self-attn)
        x_1m_proj = self.input_projection(x_1m)
        h_1m_seq = self.transformer_1m(x_1m_proj, x_1m_proj)
        h_1m = h_1m_seq[:, -1, :]  # take last token

        fused = torch.cat([h_1d, h_1w, h_1m], dim=1)
        fused = self.feature_fusion(fused)
        out = self.fc(fused)
        return out


