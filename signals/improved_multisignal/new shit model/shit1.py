import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- small utils ----
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        # x: (..., d)
        return x * (self.w / torch.clamp(x.pow(2).mean(dim=-1, keepdim=True), min=self.eps).sqrt())

class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv1d(c, c//r, 1)
        self.fc2 = nn.Conv1d(c//r, c, 1)
    def forward(self, x):                 # x: [B, C, L]
        s = x.mean(-1, keepdim=True)      # GAP
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class DepthwiseConv1d(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding=k//2, groups=c, bias=False)
        self.pw = nn.Conv1d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm1d(c)
    def forward(self, x):                 # [B, C, L]
        return F.silu(self.bn(self.pw(self.dw(x))))

class RelativePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
    def forward(self, x):                 # x: [B, N, D]
        B, N, D = x.shape
        pe = self.encoding[:N, :].unsqueeze(0).expand(B, -1, -1)
        return x + pe

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.dw_local = DepthwiseConv1d(d_model, k=5)          # local context, lightweight
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # [B, N, D]
        # pre-norm self-attn
        xa = self.norm1(x)
        attn_out, _ = self.attn(xa, xa, xa, need_weights=False)
        x = x + attn_out
        # local depthwise conv on sequence dim
        xl = self.norm2(x)
        y = xl.transpose(1, 2)                 # [B, D, N]
        y = self.dw_local(y).transpose(1, 2)   # [B, N, D]
        x = x + y
        # FFN
        xf = self.norm3(x)
        x = x + self.ffn(xf)
        return x

class HybridBinaryModel(nn.Module):
    """
    Binary classification (defect prob per signal). ONNX-friendly.
    """
    def __init__(self, signal_length=320, hidden_sizes=[128, 64, 32],
                 num_heads=8, dropout=0.1, num_transformer_layers=4):
        super().__init__()
        Cmid = 64

        # Conv trunk (residual + SE)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1, bias=False), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Conv1d(32, Cmid, 3, padding=1, bias=False), nn.BatchNorm1d(Cmid), nn.SiLU(),
        )
        self.block = nn.Sequential(
            nn.Conv1d(Cmid, Cmid, 5, padding=2, bias=False), nn.BatchNorm1d(Cmid), nn.SiLU(),
            nn.Dropout(dropout),
            SE(Cmid, r=8),
        )

        # fixed 128 tokens via AdaptiveAvgPool (robust + ONNX-ok)
        self.token_pool = nn.AdaptiveAvgPool1d(128)  # -> [B*N, Cmid, 128]

        # project per-token features to 128 then MLP to hidden_sizes[1]
        self.proj = nn.Conv1d(Cmid, 128, 1, bias=False)
        self.shared = nn.Sequential(
            nn.Linear(128, hidden_sizes[0]), nn.Dropout(dropout), nn.GELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.Dropout(dropout), nn.GELU(),
        )

        self.pos = RelativePositionEncoding(max_len=300, d_model=hidden_sizes[1])
        self.tr_layers = nn.ModuleList([
            TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2], dropout)
            for _ in range(num_transformer_layers)
        ])

        self.pre_cls = RMSNorm(hidden_sizes[1])
        self.cls = nn.Sequential(nn.Dropout(dropout*0.5), nn.Linear(hidden_sizes[1], 1))

    def forward(self, x):                           # x: [B, N, S]
        B, N, S = x.shape
        x = x.view(B*N, 1, S)
        y = self.stem(x)
        y = y + self.block(y)                       # residual
        y = self.token_pool(y)                      # [B*N, Cmid, 128]
        y = self.proj(y)                            # [B*N, 128, 128]
        y = y.mean(dim=1)                           # GAP over channels -> [B*N, 128]
        y = self.shared(y).view(B, N, -1)           # [B, N, D]

        z = self.pos(y)
        for t in self.tr_layers:
            z = t(z)

        z = self.pre_cls(z)
        logits = self.cls(z).squeeze(-1)            # [B, N]
        prob = torch.sigmoid(logits)
        return prob
