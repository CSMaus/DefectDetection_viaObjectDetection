import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os


class SignalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.set_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                         os.path.isdir(os.path.join(root_dir, d))]
        self.signal_sets = []
        self.labels = []

        for set_dir in self.set_dirs:
            signals = []
            for filename in sorted(os.listdir(set_dir), key=lambda x: int(x.split('_')[1])):
                if filename.endswith('.txt'):
                    file_path = os.path.join(set_dir, filename)
                    signal = np.loadtxt(file_path)
                    signals.append(signal)
            self.signal_sets.append(torch.tensor(signals, dtype=torch.float32))
            has_defect = any('Defect' in filename for filename in os.listdir(set_dir))
            self.labels.append(torch.tensor([1.0 if has_defect else 0.0], dtype=torch.float32))

    def __len__(self):
        return len(self.signal_sets)

    def __getitem__(self, idx):
        return self.signal_sets[idx], self.labels[idx]


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiheadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.layer_norm(attn_output + x)


class SetTransformer(nn.Module):
    def __init__(self, signal_length, num_heads=8, dim_hidden=128, num_outputs=1):
        super(SetTransformer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(signal_length, dim_hidden),
            nn.ReLU(),
            MultiheadAttentionBlock(dim_hidden, num_heads),
            MultiheadAttentionBlock(dim_hidden, num_heads)
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(),
            nn.Linear(dim_hidden // 2, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        encoded = self.encoder(x)
        encoded = encoded.mean(dim=0)
        return self.decoder(encoded)


def load_and_predict(model_path, signal_set_path):
    signal_length = 320  # this should match signal data length
    model = SetTransformer(signal_length=signal_length, num_heads=8)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    signals = []
    for filename in sorted(os.listdir(signal_set_path), key=lambda x: int(x.split('_')[1])):
        if filename.endswith('.txt'):
            file_path = os.path.join(signal_set_path, filename)
            signal = np.loadtxt(file_path)
            signals.append(signal)

    signal_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(signal_tensor)
    return output.item()


signal_set_path = 'D:/DataSets/!!NaWooDS_toTest/WOT-D456_A4_002AscanIdx_38/'
prediction = load_and_predict('set_transformer_model.pth', signal_set_path)
print(f"Defect likelihood: {prediction:.4f}")
