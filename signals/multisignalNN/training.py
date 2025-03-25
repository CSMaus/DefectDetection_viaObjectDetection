import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os


class MultiSignalClassifier(nn.Module):
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

class SignalDataset(Dataset):
    def __init__(self, root_dir, num_signals_per_set):
        self.root_dir = root_dir
        self.set_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.signal_sets = []
        self.labels = []
        self.defect_positions = []  # Store defect start-end positions
        self.num_signals_per_set = num_signals_per_set

        for set_dir in self.set_dirs:
            signals = []
            labels = []
            defect_positions = []

            # Sorting signals based on first number in filename
            signal_files = sorted([f for f in os.listdir(set_dir) if f.endswith('.txt')],
                                  key=lambda x: int(round(float(x.split('_')[0]))))

            for filename in signal_files[:num_signals_per_set]:
                file_path = os.path.join(set_dir, filename)
                signal = np.loadtxt(file_path)
                signals.append(signal)

                defect_name = filename.split('_')[1]
                if defect_name == 'Health':
                    labels.append(0.0)
                    defect_positions.append([0, 0])  # No defect
                else:
                    # Extract defect position from filename (e.g., defect_10-20.txt -> [10, 20])
                    defect_range = filename.split('.')[0].split('_')[2].split('-')
                    defect_start, defect_end = int(defect_range[0]), int(defect_range[1])
                    labels.append(1.0)
                    defect_positions.append([defect_start, defect_end])

            self.signal_sets.append(np.array(signals, dtype=np.float32))
            self.labels.append(np.array(labels, dtype=np.float32))
            self.defect_positions.append(np.array(defect_positions, dtype=np.float32))

    def __len__(self):
        return len(self.signal_sets)

    def __getitem__(self, idx):
        signals = torch.tensor(self.signal_sets[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        defect_positions = torch.tensor(self.defect_positions[idx], dtype=torch.float32)
        return signals, labels, defect_positions


# Training Setup
signal_length = 320
hidden_sizes = [128, 64, 32]
num_signals_per_set = 300

model = MultiSignalClassifier(signal_length=signal_length, hidden_sizes=hidden_sizes)

# ds_path = "D:/DataSets/!0_0NaWooDS/2024_12_02_CollectedDS/WOT-D1_3-01_Ch-0/"
ds_path = "D:/DataSets/!0_0NaWooDS/2025_DS/2BottomRef/"
dataset = SignalDataset(ds_path, num_signals_per_set)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
history = {'epochs': [], 'train_loss': [], 'train_accuracy': []}

for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    for signals, labels, defect_positions in dataloader:
        outputs = model(signals)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += ((outputs > 0.5) == labels).sum().item()

    train_loss = total_loss / len(dataloader)
    train_accuracy = total_correct / (len(dataloader) * num_signals_per_set)

    history['epochs'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}')

# Save Model
modelname = "OPD"
torch.save(model.state_dict(), f'MultiSignalClassifier_model{modelname}.pth')

# Save Scripted Model
scripted_model = torch.jit.script(model)
scripted_model.save(f'MultiSignalClassifier_model{modelname}.pt')

# Save Training History
with open(f'MultiSignalClassifier_model{modelname}-training_history.json', 'w') as f:
    json.dump(history, f)
