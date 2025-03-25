import torch.nn as nn
import torch


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

        shifted_x = torch.cat([x[:, 1:], x[:, -1:].detach()], dim=1)  # Shift right
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

        # self.classifier = nn.Linear(hidden_sizes[1], 3)  # No activation here

        # Updated classifier to output [defect_probability, defect_start, defect_end]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1),  # to make 3 outputs - place "3" the end instead of 1
            # nn.Sigmoid()  # Ensures outputs are in range [0,1]
        )

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
        defect_prob = outputs

        # this is for 3 output predictions
        # defect_prob = torch.sigmoid(outputs[:, :, 0])
        # defect_start = torch.tanh(outputs[:, :, 1]) * 0.5 + 0.5
        # defect_end = torch.tanh(outputs[:, :, 2]) * 0.5 + 0.5
        # return defect_prob, defect_start, defect_end

        return defect_prob.squeeze(-1)






class MultiSignalClassifier_OLD(nn.Module):
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
