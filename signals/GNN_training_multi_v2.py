import json
import sys

import torch
import torch.nn as nn
import numpy as np
from graphviz import Digraph
from torch.utils.data import DataLoader, Dataset
import os


class MultiSignalClassifier(nn.Module):
    def __init__(self, signal_length, hidden_sizes):
        super(MultiSignalClassifier, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, hidden_sizes[0]),  # , 3),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),  # 3),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(hidden_sizes[1], num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),  # , 3),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1),  # 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        x = x.view(batch_size * num_signals, signal_length)
        shared_out = self.shared_layer(x)
        shared_out = shared_out.view(batch_size, num_signals, -1)

        attn_output, _ = self.attention(shared_out, shared_out, shared_out)

        outputs = self.classifier(attn_output)  # Output prediction for each signal
        return outputs.squeeze(-1)


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
                                  key=lambda x: int(round(float(x.split('_')[0]))))  # [1] is type of defect

            for filename in signal_files[:num_signals_per_set]:
                file_path = os.path.join(set_dir, filename)
                signal = np.loadtxt(file_path)
                signals.append(signal)
                defect_name = filename.split('_')[1]
                # if 'Health' in filename:
                if defect_name == 'Health':
                    labels.append(0.0)
                    defect_position = [0,0]
                else:
                    defect_position = [filename.split('.')[0].split('_')[2].split('-')[0], filename.split('.')[0].split('_')[2].split('-')[1]]
                    labels.append(1.0)

            self.signal_sets.append(np.array(signals, dtype=np.float32))
            self.labels.append(np.array(labels, dtype=np.float32))

    def __len__(self):
        return len(self.signal_sets)

    def __getitem__(self, idx):
        signals = torch.tensor(self.signal_sets[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return signals, labels


def visualize_models(signal_length, hidden_sizes, filename='models_comparison'):
    dot = Digraph(comment='Model Comparison')

    dot.node('A', f'Input Layer\nSignal Length: {signal_length}')
    prev_layer = 'A'

    for i, hidden_size in enumerate(hidden_sizes):
        layer_name = f'Hidden Layer {i + 1}\nUnits: {hidden_size}'
        activation_name = f'ReLU {i + 1}'
        dot.node(f'H{i}', layer_name)
        dot.node(f'R{i}', activation_name)
        dot.edge(prev_layer, f'H{i}')
        dot.edge(f'H{i}', f'R{i}')
        prev_layer = f'R{i}'

    dot.node('IndependentOutput', 'Output Layer\nUnits: 1\nActivation: Sigmoid')
    dot.edge(prev_layer, 'IndependentOutput')

    # Divider between models
    dot.node('Divider', '', shape='plaintext')

    # Multi-Signal Processing Model with Attention
    dot.node('Input', f'Input Layer\nSignal Length: {signal_length}\nNum Signals: Variable')
    prev_layer = 'Input'

    for i, hidden_size in enumerate(hidden_sizes[:-1]):  # Skip last layer size for attention focus
        layer_name = f'Hidden Layer {i + 1}\nUnits: {hidden_size}'
        activation_name = f'ReLU {i + 1}'
        dot.node(f'MH{i}', layer_name)
        dot.node(f'MR{i}', activation_name)
        dot.edge(prev_layer, f'MH{i}')
        dot.edge(f'MH{i}', f'MR{i}')
        prev_layer = f'MR{i}'

    dot.node('Attention', 'Attention Layer\nAggregates Features')
    dot.edge(prev_layer, 'Attention')

    dot.node('Context', 'Context Vector\nInfluence of Signals')
    dot.edge('Attention', 'Context')

    dot.node('MultiSignalOutput', 'Output Layer\nUnits: 1\nPer Signal\nActivation: Sigmoid')
    dot.edge('Context', 'MultiSignalOutput')

    dot.render(filename, format='png', cleanup=True)
    print(f"Diagram saved as {filename}.png")


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
hidden_sizes = [128, 64, 32]
num_signals_per_set = 300

model = MultiSignalClassifier(signal_length=signal_length, hidden_sizes=hidden_sizes)
visualize_models(signal_length, hidden_sizes, filename='models_comparison')

ds_path = "D:/DataSets/!0_0NaWooDS/2024_12_02_CollectedDS/WOT-D1_3-01_Ch-0/"  # "D:/DataSets/!0_0NaWooDS/FPD_D456/"  # 'D:/DataSets/!!NaWooDS/'
dataset = SignalDataset(ds_path, num_signals_per_set)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
history = {'epochs': [], 'train_loss': [], 'train_accuracy': []}

for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    for signals, labels in dataloader:
        outputs = model(signals)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += ((outputs > 0.5) == labels).sum().item()

    train_loss = total_loss / len(dataloader)
    train_accuracy = total_correct / (len(dataloader) * num_signals_per_set)

    update_training_history(history, epoch, train_loss, train_accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

modelname = "OPD"  # "FPD"
torch.save(model.state_dict(), f'MultiSignalClassifier_model{modelname}.pth')

date = '2024_12_03'
scripted_model = torch.jit.script(model)
scripted_model.save(f'MultiSignalClassifier_model{modelname}.pt')

save_training_history(history, filename=f'MultiSignalClassifier_model{modelname}-training_history.json')

