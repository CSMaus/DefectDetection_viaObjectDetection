import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        batch_size, num_signals, hidden_dim = x.shape
        position_encoding = self.encoding[:num_signals, :].unsqueeze(0).expand(batch_size, -1, -1)
        return x + position_encoding


class LocalAttention(nn.Module):
    """
    Local attention using convolutional layers to focus on neighboring signals.
    Increased kernel size for wider context window.
    """

    def __init__(self, d_model, kernel_size=11):
        # Increased from 5 to kernel 11 for wider context
        super().__init__()
        self.local_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size,
                                    padding=kernel_size // 2, groups=d_model)

        # self.alpha = nn.Parameter(torch.tensor(0.5))  # may help for local context

    def forward(self, x):
        # (B, N, d_model) -> (B, d_model, N) -> (B, N, d_model)
        # x = x - x.mean(dim=1, keepdim=True)  # should I remove mean to "reduce" background?
        x = x.permute(0, 2, 1)
        x = self.local_conv(x)
        x = x.permute(0, 2, 1)

        # y = self.local_conv(x.permute(0,2,1)).permute(0,2,1)
        #         return x + self.alpha * y

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.15):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.self_attn = TinyMHA(d_model, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.local_attn = LocalAttention(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # attn_out, _ = self.self_attn(x, x, x) bcs now not nn.MultiheadAttention
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        # xa = self.norm1(x)
        # attn_out, _ = self.self_attn(xa, xa, xa, need_weights=False)
        # x = x + self.dropout(attn_out)

        local_attn_out = self.local_attn(x)
        x = self.norm2(x + self.dropout(local_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x

class TinyMHA(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, bias=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.o = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # x: [B, N, D]
        B, N, D = x.shape
        q = self.q(x).view(B, -1, self.h, self.dh).transpose(1, 2)   # [B,h,N,dh]
        k = self.k(x).view(B, -1, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, -1, self.h, self.dh).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)         # [B,h,N,N]
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v                                                # [B,h,N,dh]
        y = y.transpose(1, 2).contiguous().view(B, -1, D)            # [B,N,D]
        y = self.proj_drop(self.o(y))                               # [B,N,D]
        return y




class HybridBinaryModel(nn.Module):
    """
    Hybrid model: improved_model transformer + direct_defect feature extraction
    Binary classification only (defect/no-defect)
    """

    def __init__(self, signal_length=320, hidden_sizes=[256, 192, 48], num_heads=8, dropout=0.15,
                 num_transformer_layers=4):
        super(HybridBinaryModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        kernel_size = signal_length // 128
        if kernel_size < 1:
            kernel_size = 1
        self.fixed_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)

        self.shared_layer = nn.Sequential(
            nn.Linear(256, hidden_sizes[0]),
            # increased from 128, maybe will be good for large water part signals
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        self.position_encoding = RelativePositionEncoding(max_len=1200, d_model=hidden_sizes[1])
        # need to check it to be larger so it could take very long sequences

        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2], dropout)
            for _ in range(num_transformer_layers)
        ])
        self.classifier = nn.Linear(hidden_sizes[1], 1)

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()

        x = x.view(batch_size * num_signals, 1, signal_length)
        x = self.conv_layers(x)  # (batch * num_signals, 64, signal_length)

        x = self.fixed_pool(x)  # (batch * num_signals, 64, ~128)

        current_size = x.size(2)
        # if current_size != 128:
        x = F.interpolate(x, size=128, mode='linear', align_corners=False)

        # global average pooling
        x = x.mean(dim=1)  # (batch * num_signals, 128)
        # todo: checking these 5 lines
        seq = x.view(batch_size, num_signals, -1)
        seq_mean = seq.mean(dim=1, keepdim=True).expand(-1, num_signals, -1)
        seq_concat = torch.cat([seq, seq - seq_mean], dim=-1)  # (B, N, 256)
        # # shared feature extraction
        # shared_out = self.shared_layer(x)  # todo: changed to the line below
        shared_out = self.shared_layer(seq_concat.view(batch_size * num_signals, -1))
        shared_out = shared_out.view(batch_size, num_signals, -1)

        shared_out = self.position_encoding(shared_out)
        for transformer in self.transformer_layers:
            shared_out = transformer(shared_out)

        defect_logits = self.classifier(shared_out).squeeze(-1)  # (B, N)
        defect_prob = torch.sigmoid(defect_logits)

        return defect_prob
