import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
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
                                  key=lambda x: int(round(float(x.split('_')[0]))))
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


def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            predictions = (outputs > 0.5).float()
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1


# Load saved model
signal_length = 320
num_signals_per_set = 300
modelname = "Conv1d_OPD"
model_path = f'models/MSC_model{modelname}.pth'
ds_path = "D:/DataSets/!0_0NaWooDS/2024_12_02_CollectedDS/WOT-D1_3-01_Ch-0/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DefectDetectionModel(signal_length, num_signals_per_set)
model.load_state_dict(torch.load(model_path))
model.to(device)

# Load dataset and dataloader
dataset = SignalDataset(ds_path, num_signals_per_set)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Evaluate the model
accuracy, precision, recall, f1 = evaluate_model(model, dataloader, device)

print(f"Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
