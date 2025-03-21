import json
import sys

import torch
import torch.nn as nn
import numpy as np
from graphviz import Digraph
from torch.utils.data import DataLoader, Dataset
import os


class SignalDataset(Dataset):
    def __init__(self, root_dir, num_signals_per_set):
        self.root_dir = root_dir
        self.set_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                         os.path.isdir(os.path.join(root_dir, d))]
        self.signal_sets = []
        self.labels = []
        self.num_signals_per_set = num_signals_per_set

        for set_dir in self.set_dirs:
            signals = []
            labels = []
            signal_files = sorted([f for f in os.listdir(set_dir) if f.endswith('.txt')],
                                  key=lambda x: int(round(float(x.split('_')[0]))))  # it was 1, but now [1] is type of defect
            for filename in signal_files[:num_signals_per_set]:
                file_path = os.path.join(set_dir, filename)
                signal = np.loadtxt(file_path)
                signals.append(signal)
                if 'Health' in filename:
                    labels.append(0.0)
                else:
                    labels.append(1.0)

            self.signal_sets.append(np.array(signals, dtype=np.float32))
            self.labels.append(np.array(labels, dtype=np.float32))

    def __len__(self):
        return len(self.signal_sets)

    def __getitem__(self, idx):
        signals = torch.tensor(self.signal_sets[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        signals = signals.permute(1, 0)
        return signals, labels


class DefectDetectionModel(nn.Module):
    def __init__(self, signal_length, num_signals_per_set):
        super(DefectDetectionModel, self).__init__()

        # Feature extractor for single-channel signals
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1, stride=1),
            nn.ReLU()
        )

        # Transformer encoder for sequence of signals
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True),
            num_layers=4
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, signal_length, num_signals = x.size()

        x = x.permute(0, 2, 1).contiguous().view(-1, 1, signal_length)  # [batch_size * num_signals, 1, signal_length]

        x = self.feature_extractor(x)  # [batch_size * num_signals, 128, reduced_length]
        x = x.mean(dim=2)  # Global average pooling over signal length: [batch_size * num_signals, 128]
        x = x.view(batch_size, num_signals, -1)  # [batch_size, num_signals, 128]
        x = self.transformer_encoder(x)  # [batch_size, num_signals, 128]
        x = self.classifier(x).squeeze(-1)  # [batch_size, num_signals]

        return x


def save_training_history(history, filename='hist.json'):
    with open(filename, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {filename}")


def update_training_history(history, epoch, train_loss, train_accuracy):
    history['epochs'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)
    return history


signal_length = 320  # 360  # 320
hidden_sizes = 10   # [128, 64, 32]
num_signals_per_set = 300

model = DefectDetectionModel(signal_length, num_signals_per_set)

ds_path = "D:/DataSets/!0_0NaWooDS/2024_12_02_CollectedDS/WOT-D1_3-01_Ch-0/"  # "D:/DataSets/!0_0NaWooDS/FPD_D456/"  # 'D:/DataSets/!!NaWooDS/'
dataset = SignalDataset(ds_path, num_signals_per_set)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 20
history = {'epochs': [], 'train_loss': [], 'train_accuracy': []}

for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    for signals, labels in dataloader:
        outputs = model(signals)
        loss = criterion(outputs, labels)  # Both shapes must match
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += ((outputs > 0.5) == labels).sum().item()

    train_loss = total_loss / len(dataloader)
    train_accuracy = total_correct / (len(dataloader) * num_signals_per_set)

    update_training_history(history, epoch, train_loss, train_accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

modelname = "Conv1d_OPD"  # "FPD"
torch.save(model.state_dict(), f'models/MSC_model{modelname}.pth')

date = '2024_12_03'
scripted_model = torch.jit.script(model)
scripted_model.save(f'models/MSC_model{modelname}.pt')

save_training_history(history, filename=f'training_histories/MSC_model{modelname}-training_history.json')