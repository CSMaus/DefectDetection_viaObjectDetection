import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
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


root_dir = 'D:/DataSets/!!NaWooDS/'
dataset = SignalDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

signal_length = 320  # Make sure this matches your data
model = SetTransformer(signal_length=signal_length, num_heads=8)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    total_loss = 0
    for signals, labels in dataloader:
        outputs = model(signals)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

torch.save(model.state_dict(), 'set_transformer_model.pth')
print("Model saved as 'set_transformer_model.pth'")
