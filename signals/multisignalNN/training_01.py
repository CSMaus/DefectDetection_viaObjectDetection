# NN modification to cath spatial features, train the capture background features, etc
# Fix Multihead Self-Attention Limitations
# Issue:
# Right now, the attention mechanism treats all signals equally, which does not explicitly reinforce spatial relationships.
# Defects should propagate to nearby signals (i.e., i-1, i, i+1).
# The model should be able to define "background" patterns in healthy signals and detect deviations.
# Solution:
# ✅ Use Relative Positional Encoding:
# Adds learned embeddings that encode the relative distances between signals.
# Helps the model understand that nearby signals are more important than distant ones.
# ✅ Use Cross-Attention Instead of Just Self-Attention:
# Instead of using Q=K=V (self-attention), I will make Q = current signal and K,V = nearby signals (i-1, i, i+1).
# This ensures that a signal attends more to its direct neighbors rather than all signals equally.
# ✅ Use Additional Feature Extraction with CNNs:
# 1D CNNs will be used before attention to capture local features first, ensuring patterns from healthy signals are properly extracted.
# 2️⃣ Train the Model to Recognize Background Patterns in Healthy Signals
# Issue:
# The model should learn what is "normal" (background patterns) and identify deviations as defects.
# Some defects are subtle and might blend with background noise.
# Solution:
# ✅ Contrastive Learning Approach for Background Detection
# The model should learn a "background pattern representation" from multiple healthy signals and compare defects against this learned pattern.
# Use an additional contrastive loss term that minimizes differences between healthy signals and maximizes the deviation of defective signals.
# ✅ Use a Transformer Encoder with Global Context Extraction
# Instead of just multi-head attention, I will introduce a Transformer Encoder layer, which learns both local and global patterns across multiple signals.
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.functional as F
from NN_models import RelativePositionEncoding, TransformerEncoder, MultiSignalClassifier




# Signal Dataset Class with Automatic Sequence Length Calculation
class SignalDataset(Dataset):
    def __init__(self, root_dir, root_dir2):
        self.root_dir = root_dir
        self.root_dir2 = root_dir2
        self.original_datafiles_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        self.additional_list = [os.path.join(root_dir2, d) for d in os.listdir(root_dir2)]
        self.original_datafiles_folders.extend(self.additional_list)
        #  if os.path.isdir(os.path.join(root_dir, d))
        self.signal_sets = []
        self.labels = []
        self.defect_positions = []

        self.num_signals_per_set = self._find_min_sequence_length()

        # here we list through all filefolders (folders with names of OPD files)
        for original_datafile_dir in self.original_datafiles_folders:
            beams_dirs = [os.path.join(original_datafile_dir, d) for d in os.listdir(original_datafile_dir)]
            if not beams_dirs:
                continue

            # and each filefolder has multiple folders with beams_folders, each one of them has multiple scan files
            for beam_dir in beams_dirs:
                # signal_files = [os.path.join(beam_dir, d) for d in os.listdir(beam_dir)]
                signal_files = sorted([f for f in os.listdir(beam_dir) if f.endswith('.txt')],
                                      key=lambda x: int(round(float(x.split('_')[0]))))
                if not signal_files:
                    continue



                # for beam_scans_dir in beam_scans_dirs:
                    # signal_files = sorted([f for f in os.listdir(beam_dir) if f.endswith('.txt')],
                    #                       key=lambda x: int(round(float(x.split('_')[0]))))
                signals = []
                labels = []
                defect_positions = []

                for filename in signal_files[:self.num_signals_per_set]:
                    file_path = os.path.join(beam_dir, filename)
                    signal = np.loadtxt(file_path)
                    signals.append(signal)

                    defect_name = filename.split('.')[0].split('_')[1]
                    if defect_name == 'Health':
                        labels.append(0.0)
                        defect_positions.append([0.0, 0.0])
                    else:
                        defect_range = filename[:-4].split('_')[2].split('-')  # .split('.')[0]
                        defect_start, defect_end = float(defect_range[0]), float(defect_range[1])
                        labels.append(1.0)
                        defect_positions.append([defect_start, defect_end])

                self.signal_sets.append(np.array(signals, dtype=np.float32))
                self.labels.append(np.array(labels, dtype=np.float32))
                self.defect_positions.append(np.array(defect_positions, dtype=np.float32))

    def _find_min_sequence_length(self):
        min_lengths = []
        for filefolder in self.original_datafiles_folders:
            beam_0_path = os.path.join(filefolder, os.listdir(filefolder)[0])
            num_files = len(os.listdir(beam_0_path))
            min_lengths.append(num_files)
        return min(min_lengths)

    def __len__(self):
        return len(self.signal_sets)

    def __getitem__(self, idx):
        signals = torch.tensor(self.signal_sets[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        defect_positions = torch.tensor(self.defect_positions[idx], dtype=torch.float32)
        return signals, labels, defect_positions


signal_length = 320
num_epochs = 40
hidden_sizes = [128, 64, 32]
num_heads = 4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiSignalClassifier(signal_length=signal_length, hidden_sizes=hidden_sizes, num_heads=num_heads).to(device)
print(f"Using device: {device}")

ds_path = "D:/DataSets/!0_0NaWooDS/2025_DS/2BottomRef/"  # 2BottomRef/"  # train
ds_path2 = "D:/DataSets/!0_0NaWooDS/2025_DS/1BottomRef/"  # 2BottomRef/"  # train
dataset = SignalDataset(ds_path, ds_path2)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # with true need to reset number of samples
print("Number of signal_sequences: ", len(dataset.labels))
print("Sequence len was set to: ", dataset.num_signals_per_set)


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


history = {'epochs': [], 'train_loss': [], 'train_accuracy': []}
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    for signals, labels, defect_positions in dataloader:
        signals, labels, defect_positions = signals.to(device), labels.to(device), defect_positions.to(device)
        defect_prob_pred, defect_start_pred, defect_end_pred = model(signals)
        # defect_prob_pred = model(signals)

        # loss_classification = criterion(torch.sigmoid(defect_prob_pred), labels.float())
        loss_classification = criterion(defect_prob_pred, labels)
        '''loss_start = F.mse_loss(defect_start_pred, defect_positions[:, :, 0])
        loss_end = F.mse_loss(defect_end_pred, defect_positions[:, :, 1])'''

        defect_start_pred = torch.clamp(defect_start_pred, 0, 1)
        defect_end_pred = torch.clamp(defect_end_pred, 0, 1)
        loss_start = F.mse_loss(defect_start_pred, defect_positions[:, :, 0])
        loss_end = F.mse_loss(defect_end_pred, defect_positions[:, :, 1])

        loss = loss_classification + 0.5 * (loss_start + loss_end)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += ((defect_prob_pred > 0.5) == labels).sum().item()

    train_loss = total_loss / len(dataloader)
    train_accuracy = total_correct / (len(dataloader) * dataset.num_signals_per_set)

    history['epochs'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}')


modelname = "OPD"
attempt = "006"  # "test"  "004"
print(f"Trained model for {num_epochs} epochs, attempt: {attempt}")
torch.save(model.state_dict(), f'models/{attempt}-MultiSignalClassifier_model{modelname}.pth')

scripted_model = torch.jit.script(model)
scripted_model.save(f'models/{attempt}-MultiSignalClassifier_model{modelname}.pt')

with open(f'histories/{attempt}-MultiSignalClassifier_model{modelname}-training_history.json', 'w') as f:
    json.dump(history, f)





