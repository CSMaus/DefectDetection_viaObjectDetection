import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from graphviz import Digraph


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
                                  key=lambda x: int(x.split('_')[1]))
            for filename in signal_files[:num_signals_per_set]:
                file_path = os.path.join(set_dir, filename)
                signal = np.loadtxt(file_path)
                signals.append(signal)
                if 'Defect' in filename:
                    labels.append(1.0)
                else:
                    labels.append(0.0)

            self.signal_sets.append(torch.tensor(signals, dtype=torch.float32))
            self.labels.append(torch.tensor(labels, dtype=torch.float32))

    def __len__(self):
        return len(self.signal_sets)

    def __getitem__(self, idx):
        return self.signal_sets[idx], self.labels[idx]


class SignalClassifier(nn.Module):
    def __init__(self, signal_length, num_signals):
        super(SignalClassifier, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, 128),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        x = x.view(batch_size * num_signals, signal_length)
        out = self.shared_layer(x)
        return out.view(batch_size, num_signals)


root_dir = 'D:/DataSets/!!NaWooDS/'
num_signals_per_set = 298  # number of signals per set (should be same as in the training)
signal_length = 320  # this should match signal data length

dataset = SignalDataset(root_dir, num_signals_per_set)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = SignalClassifier(signal_length=signal_length, num_signals=num_signals_per_set)
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

torch.save(model.state_dict(), 'signal_classifier_model2.pth')
print("Model saved as 'signal_classifier_model2.pth'")
