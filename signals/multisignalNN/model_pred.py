from NN_models import MultiSignalClassifier, MultiSignalClassifier_N
import torch
import numpy as np
import os
from matplotlib import pyplot as plt


def load_and_predict(model_path, signal_set_path, num_signals_per_set, signal_length, device, _hidden_sizes=[128, 64, 32], heads=4):
    # model = MultiSignalClassifier(signal_length, hidden_sizes=[128, 64, 32]).to(device)
    # model = MultiSignalClassifier(signal_length=signal_length, hidden_sizes=_hidden_sizes, num_heads=heads).to(device)
    model = MultiSignalClassifier_N(signal_length=signal_length, hidden_sizes=_hidden_sizes, num_heads=heads).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    signals = []
    signal_files = sorted([f for f in os.listdir(signal_set_path) if f.endswith('.txt')],
                          key=lambda x: int(x.split('_')[0]))

    for filename in signal_files[:num_signals_per_set]:
        signal = np.loadtxt(os.path.join(signal_set_path, filename))
        signals.append(signal)

    signal_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        defect_prob, defect_start, defect_end = model(signal_tensor)

    return defect_prob.squeeze().cpu().numpy(), defect_start.squeeze().cpu().numpy(), defect_end.squeeze().cpu().numpy()

def load_and_predict_probsOnly(model_path, signal_set_path, num_signals_per_set, signal_length, device, _hidden_sizes=[128, 64, 32], heads=4):
    model = MultiSignalClassifier(signal_length=signal_length, hidden_sizes=_hidden_sizes, num_heads=heads).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    signals = []
    signal_files = sorted([f for f in os.listdir(signal_set_path) if f.endswith('.txt')],
                          key=lambda x: int(x.split('_')[0]))

    for filename in signal_files[:num_signals_per_set]:
        signal = np.loadtxt(os.path.join(signal_set_path, filename))
        signals.append(signal)

    signal_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(signal_tensor)

    # bcs in training applied sigmoid in losses in training loop
    outputs = torch.sigmoid(outputs)

    return outputs.squeeze().tolist()

def plot_predictions(defect_probs, defect_starts=None, defect_ends=None):
    plt.figure(figsize=(14, 6))
    plt.plot(defect_probs * 100, marker='o', linestyle='-', color='blue', label='Defect Probability (%)')
    if defect_starts is not None:
        plt.plot(defect_starts * 100, marker='x', linestyle='--', color='green', label='Defect Start Position (%)')
    if defect_ends is not None:
        plt.plot(defect_ends * 100, marker='x', linestyle='--', color='red', label='Defect End Position (%)')
    plt.xlabel('Signal Index', fontsize=12)
    plt.ylabel('Value (%)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
signal_length = 320
num_signals_per_set = 170

# signal_set_path = 'D:/DataSets/!0_0NaWooDS/2025_DS/train/787-225_01_Ch-0/BeamIdx_43/'
signal_set_path = 'D:/DataSets/!0_0NaWooDS/2025_DS/2BottomRef  - 787-404_07_Ch-0/BeamIdx_18/'
attempt = "008"
model_path = f'models/{attempt}-_N-MultiSignalClassifier_modelOPD.pth'

hidden_sizes = [128, 64, 32]
num_heads = 8 # 4
defect_probs, defect_starts, defect_ends = load_and_predict(model_path, signal_set_path, num_signals_per_set, signal_length, device)
# defect_probs = load_and_predict_probsOnly(model_path, signal_set_path, num_signals_per_set, signal_length, device, hidden_sizes, num_heads)


for i, (prob, start, end) in enumerate(zip(defect_probs, defect_starts, defect_ends)):
# for i, prob in enumerate(defect_probs):
    status = 'Defective' if prob > 0.5 else 'Healthy'
    print(f"Signal {i}: {status} | Confidence: {prob*100:.2f}% | Start: {start:.3f} | End: {end:.3f}")

print(f"model predictions for attempt: {attempt}, model _N")
plot_predictions(defect_probs, defect_starts, defect_ends)
