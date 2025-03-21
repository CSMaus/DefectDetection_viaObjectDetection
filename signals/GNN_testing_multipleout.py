import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import os


class SignalClassifier(nn.Module):
    def __init__(self, signal_length, num_signals):
        super(SignalClassifier, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(signal_length, 64),
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


def load_and_predict(model_path, signal_set_path, num_signals_per_set):
    signal_length = 320  # this should match signal data length
    model = SignalClassifier(signal_length=signal_length, num_signals=num_signals_per_set)
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


signal_set_path = 'D:/DataSets/!!NaWooDS_toTest/WOT-D456_A4_005AscanIdx_16/'  # WOT-D456_A4_002AscanIdx_38/'

num_signals_per_set = os.listdir(os.path.join(signal_set_path)).__len__()
#     2  # number of signals per set: the number of signals in set could be different from training

predictions = load_and_predict('signal_classifier_model2.pth', signal_set_path, num_signals_per_set)
for i, pred in enumerate(predictions):
    print(f"Signal {i}: {'Defective' if pred > 0.5 else 'Healthy'} (Confidence: {pred:.4f})")
