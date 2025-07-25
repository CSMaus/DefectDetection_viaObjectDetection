import torch.nn as nn
import torch

# Now I want you to check all newly created files, compare them with old model MultiSignalClassifier_N and check the training process - it shoud be same as in training_01.py. Also, note that even if we are loading data from Json file and not from folders, the structure of dataset should be same

class RelativePositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        batch_size, num_signals, hidden_dim = x.shape
        position_encoding = self.encoding[:num_signals, :].unsqueeze(0).expand(batch_size, -1, -1)
        return x + position_encoding

# Transformer Encoder with Cross-Attention
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        shifted_x = torch.cat([x[:, 1:], x[:, -1:].detach()], dim=1)  # Shift right signals by one step is too rigid!
        cross_attn_out, _ = self.cross_attn(x, shifted_x, shifted_x)
        x = self.norm2(x + cross_attn_out)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x

# Multi-Signal Classifier with Relative Positioning & Transformer Encoder
class MultiSignalClassifier(nn.Module):
    def __init__(self, signal_length, hidden_sizes, num_heads=4):
        super(MultiSignalClassifier, self).__init__()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )

        self.position_encoding = RelativePositionEncoding(max_len=300, d_model=hidden_sizes[1])
        self.transformer_encoder = TransformerEncoder(hidden_sizes[1], num_heads, hidden_sizes[2])

        self.classifier = nn.Linear(hidden_sizes[1], 3)  # No activation here

        # Updated classifier to output [defect_probability, defect_start, defect_end]
        '''self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 3),  # to make 3 outputs - place "3" the end instead of 1
            # nn.Sigmoid()  # Ensures outputs are in range [0,1]
        )'''

    """def shit_forward(self, x):
        batch_size, num_signals, signal_length = x.size()

        # # x = x.view(batch_size * num_signals, 1, signal_length)  # this gave error before
        # x = x.view(batch_size * num_signals, signal_length)
        # x = self.conv1d(x)
        # # x = x.view(batch_size * num_signals, signal_length)   # this gave error before
        # x = x.view(batch_size * num_signals, -1)

        x = x.view(batch_size * num_signals, 1, signal_length)
        x = self.conv1d(x)
        x = x.view(batch_size * num_signals, -1)

        shared_out = self.shared_layer(x)
        shared_out = shared_out.view(batch_size, num_signals, -1)

        shared_out = self.position_encoding(shared_out)
        shared_out = self.transformer_encoder(shared_out)

        '''outputs = self.classifier(shared_out)  # Now outputs 3 values per signal
        defect_prob = outputs[:, :, 0]  # Classification output
        defect_start = outputs[:, :, 1]  # Defect start (0-1)
        defect_end = outputs[:, :, 2]  # Defect end (0-1)'''

        outputs = self.classifier(shared_out)  # No activation in classifier
        # Apply Sigmoid only to defect_prob
        defect_prob = torch.sigmoid(outputs[:, :, 0])  # Binary classification
        defect_start = torch.tanh(outputs[:, :, 1]) * 0.5 + 0.5  # outputs[:, :, 1]  # Direct regression output
        defect_end = torch.tanh(outputs[:, :, 2]) * 0.5 + 0.5  # outputs[:, :, 2]  # Direct regression output

        return defect_prob, defect_start, defect_end"""

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()

        x = x.view(batch_size * num_signals, 1, signal_length)
        x = self.conv1d(x)

        # Correct shape fix:
        x = x.mean(dim=1)  # shape becomes (batch_size*num_signals, signal_length)

        shared_out = self.shared_layer(x)
        shared_out = shared_out.view(batch_size, num_signals, -1)

        shared_out = self.position_encoding(shared_out)
        shared_out = self.transformer_encoder(shared_out)

        outputs = self.classifier(shared_out)

        defect_prob = torch.sigmoid(outputs[:, :, 0])
        defect_start = torch.tanh(outputs[:, :, 1]) * 0.5 + 0.5
        defect_end = torch.tanh(outputs[:, :, 2]) * 0.5 + 0.5
        return defect_prob, defect_start, defect_end

        # return defect_prob.squeeze(-1)

# ################## ---------------- New version ----------------  #################################
# Replace Cross-Attention with Local Attention
# Add Relative Positional Bias to Multi-Head Attention
# Before passing signals to the Transformer, compute a running average or learnable background normalization
# to capture long-term trends
import torch.nn.functional as F


class RelativePositionEncoding_N(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        batch_size, num_signals, hidden_dim = x.shape
        position_encoding = self.encoding[:num_signals, :].unsqueeze(0).expand(batch_size, -1, -1)
        return x + position_encoding


class LocalAttention_N(nn.Module):
    """
    This replaces standard attention with local attention.
    It uses a convolutional layer to mix only neighboring signals instead of all signals.
    """

    def __init__(self, d_model, kernel_size=5):
        super().__init__()
        self.local_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size,
                                    padding=kernel_size // 2, groups=d_model)

    def forward(self, x):
        # (batch, num_signals, d_model) -> (batch, d_model, num_signals)
        x = x.permute(0, 2, 1)
        x = self.local_conv(x)  # Apply depth-wise convolution
        x = x.permute(0, 2, 1)  # Back to (batch, num_signals, d_model)
        return x


class TransformerEncoder_N(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.local_attn = LocalAttention_N(d_model)  # Local convolution-based attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Replace cross-attention shift with learnable local attention
        local_attn_out = self.local_attn(x)
        x = self.norm2(x + local_attn_out)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


class MultiSignalClassifier_N(nn.Module):
    def __init__(self, signal_length, hidden_sizes, num_heads=4):
        super(MultiSignalClassifier_N, self).__init__()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Background trend extraction
        self.background_extractor = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=11, padding=5, stride=1,
                                              groups=16)

        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )

        self.position_encoding = RelativePositionEncoding_N(max_len=300, d_model=hidden_sizes[1])
        self.transformer_encoder = TransformerEncoder_N(hidden_sizes[1], num_heads, hidden_sizes[2])

        self.classifier = nn.Linear(hidden_sizes[1], 3)

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        x = x.view(batch_size * num_signals, 1, signal_length)
        x = self.conv1d(x)

        # Extract background trend
        bg_trend = self.background_extractor(x)
        x = x - bg_trend  # Normalize signal based on extracted background

        x = x.mean(dim=1)  # shape becomes (batch_size*num_signals, signal_length)
        shared_out = self.shared_layer(x)
        shared_out = shared_out.view(batch_size, num_signals, -1)

        shared_out = self.position_encoding(shared_out)
        shared_out = self.transformer_encoder(shared_out)

        outputs = self.classifier(shared_out)
        defect_prob = torch.sigmoid(outputs[:, :, 0])
        defect_start = torch.tanh(outputs[:, :, 1]) * 0.5 + 0.5
        defect_end = torch.tanh(outputs[:, :, 2]) * 0.5 + 0.5

        return defect_prob, defect_start, defect_end


class NOTTT_MultiSignalClassifier_OLD(nn.Module):
    def __init__(self, signal_length, hidden_sizes, num_heads=4):
        super(MultiSignalClassifier, self).__init__()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Shared fully connected layers
        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )

        # Stacked Attention layers for better global dependencies
        self.attention1 = nn.MultiheadAttention(hidden_sizes[1], num_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(hidden_sizes[1], num_heads, batch_first=True)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()

        # 1D Convolution to capture local dependencies
        x = x.view(batch_size * num_signals, 1, signal_length)
        x = self.conv1d(x)
        x = x.view(batch_size * num_signals, signal_length)

        # Shared feature extraction
        shared_out = self.shared_layer(x)
        shared_out = shared_out.view(batch_size, num_signals, -1)

        # Attention Layers
        attn_output, _ = self.attention1(shared_out, shared_out, shared_out)
        attn_output, _ = self.attention2(attn_output, attn_output, attn_output)

        # Classification
        outputs = self.classifier(attn_output)  # Per-signal classification
        return outputs.squeeze(-1)  # Shape: (batch_size, num_signals)
