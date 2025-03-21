import torch
import numpy as np
import torch.nn as nn
import os
from matplotlib import pyplot as plt

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

    return outputs.squeeze().tolist(), signals

def calculate_reference_signal(predictions, signals):
    healthy_signals = [signal for pred, signal in zip(predictions, signals) if pred < 0.5]
    if not healthy_signals:
        return None
    reference_signal = np.mean(healthy_signals, axis=0)
    return reference_signal

def calculate_difference_matrix(reference_signal, predictions, signals):
    difference_matrix = []
    for pred, signal in zip(predictions, signals):
        if pred >= 0.5:  # Only calculate difference for defective signals
            difference = np.abs(signal - reference_signal)
        else:
            difference = np.zeros_like(signal)  # Zero vector for healthy signals
        difference_matrix.append(difference)
    return np.array(difference_matrix)

def plot_difference_heatmap(difference_matrix, font_type='Arial', font_size=12):
    plt.figure(figsize=(12, 8))
    plt.imshow(difference_matrix.T, cmap='hot', interpolation='nearest', aspect='auto')

    cbar = plt.colorbar()
    cbar.set_label('Difference with reference signal', fontsize=font_size, fontname=font_type)
    # plt.title('Difference Heatmap (Defective Signals vs. Reference)', fontsize=font_size + 2, fontname=font_type)
    plt.xlabel('Scan, mm', fontsize=font_size, fontname=font_type)
    plt.ylabel('Depth, mm', fontsize=font_size, fontname=font_type)
    plt.gca().set_yticks(np.arange(0, 321, 40))
    plt.gca().set_yticklabels([f'{int(tick / 10)}' for tick in np.arange(0, 321, 40)], fontsize=font_size,
                              fontname=font_type)
    plt.xticks(fontsize=font_size, fontname=font_type)
    plt.yticks(fontsize=font_size, fontname=font_type)
    plt.show()

num_signals_per_set = 298
signal_length = 320
signal_set_path = 'D:/DataSets/!!NaWooDS_toTest/WOT-D456_A4_002AscanIdx_38/'

predictions, signals = load_and_predict('MultiSignalClassifier_model2.pth', signal_set_path, num_signals_per_set, signal_length)

reference_signal = calculate_reference_signal(predictions, signals)
if reference_signal is not None:
    difference_matrix = calculate_difference_matrix(reference_signal, predictions, signals)
    plot_difference_heatmap(difference_matrix, font_type='Times New Roman', font_size=15)
else:
    print("No healthy signals found to calculate the reference signal.")

# for i, pred in enumerate(predictions):
#     print(f"Signal {i}: {'Defective' if pred > 0.5 else 'Healthy'} (Confidence: {pred * 100:.2f}%)")
