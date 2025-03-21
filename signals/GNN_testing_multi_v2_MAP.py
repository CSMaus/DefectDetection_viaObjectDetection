import torch
import numpy as np
import torch.nn as nn
import os
from matplotlib import pyplot as plt
import re


class MultiSignalClassifier(nn.Module):
    def __init__(self, signal_length, hidden_sizes):
        super(MultiSignalClassifier, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(hidden_sizes[1], num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_signals, signal_length = x.size()
        x = x.view(batch_size * num_signals, signal_length)
        shared_out = self.shared_layer(x)
        shared_out = shared_out.view(batch_size, num_signals, -1)

        attn_output, _ = self.attention(shared_out, shared_out, shared_out)

        outputs = self.classifier(attn_output)
        return outputs.squeeze(-1)


def load_and_predict(model_path, signal_set_path, num_signals_per_set, signal_length):
    model = MultiSignalClassifier(signal_length=signal_length, hidden_sizes=[128, 64, 32])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    signals = []
    signal_files = sorted([f for f in os.listdir(signal_set_path) if f.endswith('.txt')],
                          key=lambda x: int(x.split('_')[1]))
    for filename in signal_files[:num_signals_per_set]:
        file_path = os.path.join(signal_set_path, filename)
        signal = np.loadtxt(file_path)
        signals.append(signal)

    signal_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = model(signal_tensor)

    return outputs.squeeze().tolist()


def generate_prediction_map(base_path, model_path, num_signals_per_set, signal_length):
    all_predictions = []
    idx_folders = sorted([d for d in os.listdir(base_path) if re.match(r'WOT-D456_A4_001+AscanIdx_\d+', d)])

    for folder in idx_folders:
        folder_path = os.path.join(base_path, folder)
        predictions = load_and_predict(model_path, folder_path, num_signals_per_set, signal_length)
        all_predictions.append(predictions)

    return np.array(all_predictions)


def plot_prediction_heatmap(prediction_map, font_type='Arial', font_size=12):
    plt.figure(figsize=(10, 8))
    plt.imshow(prediction_map * 100, cmap='coolwarm', interpolation='spline16', aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Defective Probability (%)', fontsize=font_size, fontname=font_type)

    # plt.title('Prediction Heatmap', fontsize=font_size + 2, fontname=font_type)
    plt.xlabel('Scan Index', fontsize=font_size, fontname=font_type)
    plt.ylabel('Beam Index', fontsize=font_size, fontname=font_type)
    plt.xticks(fontsize=font_size, fontname=font_type)
    plt.yticks(fontsize=font_size, fontname=font_type)
    plt.show()


# Parameters
num_signals_per_set = 298
signal_length = 320
base_path = 'D:/DataSets/!!NaWooDS/'
model_path = 'MultiSignalClassifier_model3.pth'
prediction_map = generate_prediction_map(base_path, model_path, num_signals_per_set, signal_length)

plot_prediction_heatmap(prediction_map, font_type='Times New Roman', font_size=16)
