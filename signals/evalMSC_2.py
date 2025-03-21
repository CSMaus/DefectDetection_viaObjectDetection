import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os


class SignalDatasetSingleFolder(Dataset):
    def __init__(self, folder_path, signal_length):
        self.folder_path = folder_path
        self.signal_length = signal_length
        self.signals = []
        self.labels = []

        signal_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')],
                              key=lambda x: int(round(float(x.split('_')[0]))))

        for filename in signal_files:
            file_path = os.path.join(folder_path, filename)
            signal = np.loadtxt(file_path)
            self.signals.append(signal)
            if 'Health' in filename:
                self.labels.append(0.0)
            else:
                self.labels.append(1.0)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return signal, label


class DefectDetectionModel(nn.Module):
    def __init__(self, signal_length, num_signals_per_set):
        super(DefectDetectionModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True),
            num_layers=4
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, signal_length, num_signals = x.size()
        x = x.permute(0, 2, 1).contiguous().view(-1, 1, signal_length)
        x = self.feature_extractor(x)
        x = x.mean(dim=2)
        x = x.view(batch_size, num_signals, -1)
        x = self.transformer_encoder(x)
        x = self.classifier(x).squeeze(-1)
        return x


# Load the saved model
signal_length = 320
num_signals_per_set = 300  # Not used here, but kept for consistency with the model
modelname = "Conv1d_OPD"
model_path = f'models/MSC_model{modelname}.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DefectDetectionModel(signal_length, num_signals_per_set)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Load data from the specific folder
folder_path = r"D:\DataSets\!0_0NaWooDS\2024_12_02_CollectedDS\WOT-D1_3-01_Ch-0\BeamIdx_27"
dataset = SignalDatasetSingleFolder(folder_path, signal_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Generate predictions
print("Predictions and True Labels:")
for signals, labels in dataloader:
    signals = signals.unsqueeze(0).to(device)  # Add batch dimension
    labels = labels.to(device)
    outputs = model(signals)
    predictions = (outputs > 0.5).float()

    for i in range(signals.shape[1]):
        print(f"Signal {i + 1}: Prediction = {predictions[0][i].item()}, True Label = {labels[i].item()}")
