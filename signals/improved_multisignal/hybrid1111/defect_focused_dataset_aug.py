# defect_focused_dataset_aug.py
import torch
import numpy as np
import json
import os
import math
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, random_split


# -----------------------------
# Helpers
# -----------------------------

def _resample_1d(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Resample 1D numpy array to target_len using linear interpolation (endpoint=False)."""
    L = signal.shape[0]
    if L == target_len:
        return signal.astype(np.float32, copy=False)
    src = np.linspace(0.0, 1.0, L, endpoint=False, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, target_len, endpoint=False, dtype=np.float64)
    out = np.interp(dst, src, signal.astype(np.float64, copy=False)).astype(np.float32)
    return out


def _make_pad_values(pad_len: int, mode: str = "zeros", near_zero_range: Tuple[float, float] = (0.0, 0.02)) -> np.ndarray:
    if pad_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if mode == "near_zero":
        low, high = near_zero_range
        return np.random.uniform(low=low, high=high, size=pad_len).astype(np.float32)
    return np.zeros(pad_len, dtype=np.float32)


def _scale_positions_norm(start: float, end: float, scale: float) -> Tuple[float, float]:
    if start == 0.0 and end == 0.0:
        return 0.0, 0.0
    s = max(0.0, min(1.0, start * scale))
    e = max(0.0, min(1.0, end * scale))
    # ensure non-decreasing
    if e < s:
        s, e = e, s
    return s, e


# -----------------------------
# Dataset with augmentation
# -----------------------------

class DefectFocusedJsonSignalDataset(Dataset):
    """
    Loads sequences from JSON files. Signals are assumed normalized to [0,1] and length=320.
    Augmentations:
      1) uniform padding per sequence (same pad for all signals in the sequence), then resample to 320.
      2) variable padding across the sequence (pad increases linearly from pad_start to pad_end per signal), then resample to 320.
    Defect positions are expected to be normalized to [0,1] relative to the signal length; they are scaled accordingly.
    """
    def __init__(
        self,
        json_dir: str,
        seq_length: int = 50,
        min_defects_per_sequence: int = 1,
        isOnlyDefective: bool = False,
        augment_uniform_pad_lengths: Optional[List[int]] = None,
        augment_variable_pad_schedules: Optional[List[Tuple[int, int]]] = None,
        pad_mode: str = "zeros",  # "zeros" | "near_zero"
        near_zero_range: Tuple[float, float] = (0.0, 0.02),
    ):
        self.json_dir = json_dir
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.seq_length = seq_length
        self.min_defects_per_sequence = min_defects_per_sequence
        self.isOnlyDefective = isOnlyDefective

        self.augment_uniform_pad_lengths = augment_uniform_pad_lengths or []
        self.augment_variable_pad_schedules = augment_variable_pad_schedules or []
        self.pad_mode = pad_mode
        self.near_zero_range = near_zero_range

        self.signal_sets: List[np.ndarray] = []   # [N, 320]
        self.labels: List[np.ndarray] = []        # [N]
        self.defect_positions: List[np.ndarray] = []  # [N, 2] normalized [0,1]

        self._load_defect_sequences()

        print(f"Loaded {len(self.signal_sets)} DEFECT-CONTAINING sequences from {len(self.json_files)} JSON files")
        print(f"Each sequence contains {self.seq_length} signals")
        print(f"Minimum defects per sequence: {self.min_defects_per_sequence}")

    # -------------------------
    # Core loading
    # -------------------------
    def _load_defect_sequences(self):
        total_beams = 0
        total_sequences = 0
        total_sequences_with_defects = 0
        total_sequences_nodefects_added = 0
        total_sequences_skipped = 0

        print(f"Found {len(self.json_files)} JSON files in {self.json_dir}")

        for json_file in self.json_files:
            file_path = os.path.join(self.json_dir, json_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                for beam_key in data.keys():
                    beam_data = data[beam_key]
                    total_beams += 1

                    scans_keys_sorted = sorted(beam_data.keys(), key=lambda x: int(x.split('_')[0]))
                    if len(scans_keys_sorted) < self.seq_length:
                        continue

                    all_scans_for_beam = {}
                    all_lbls_for_beam = {}
                    all_defects_for_beam = {}

                    scan_idx = 0
                    for scan_key in scans_keys_sorted:
                        scan_data = beam_data[scan_key]
                        all_scans_for_beam[str(scan_idx)] = scan_data

                        if scan_key.split('_')[1] == "Health":
                            all_lbls_for_beam[str(scan_idx)] = 0
                            all_defects_for_beam[str(scan_idx)] = [0.0, 0.0]
                        else:
                            all_lbls_for_beam[str(scan_idx)] = 1
                            try:
                                defect_range = scan_key.split('_')[2].split('-')
                                defect_start, defect_end = float(defect_range[0]), float(defect_range[1])
                                all_defects_for_beam[str(scan_idx)] = [defect_start, defect_end]
                            except Exception:
                                all_defects_for_beam[str(scan_idx)] = [0.0, 0.0]
                        scan_idx += 1

                    num_of_seqs_for_beam = math.ceil(len(scans_keys_sorted) / self.seq_length)

                    for i in range(num_of_seqs_for_beam):
                        if i < num_of_seqs_for_beam - 1:
                            start_idx = i * self.seq_length
                            end_idx = start_idx + self.seq_length
                        else:
                            start_idx = len(scans_keys_sorted) - self.seq_length
                            end_idx = len(scans_keys_sorted)
                        if start_idx < 0:
                            continue

                        sequence = []
                        seq_labels = []
                        seq_defects = []
                        for j in range(start_idx, end_idx):
                            try:
                                scan_data = all_scans_for_beam[str(j)]
                                if isinstance(scan_data, list):
                                    signal = np.array(scan_data, dtype=np.float32)
                                elif isinstance(scan_data, dict) and 'signal' in scan_data:
                                    signal = np.array(scan_data['signal'], dtype=np.float32)
                                else:
                                    signal = np.array(scan_data, dtype=np.float32)
                                sequence.append(signal)
                                seq_labels.append(all_lbls_for_beam[str(j)])
                                seq_defects.append(all_defects_for_beam[str(j)])
                            except Exception as e:
                                print(f"Error processing scan {j} in beam {beam_key}: {e}")
                                continue

                        if len(sequence) != self.seq_length:
                            continue

                        signal_length = len(sequence[0])
                        if any(len(sig) != signal_length for sig in sequence):
                            continue

                        total_sequences += 1
                        defect_count = int(np.sum(seq_labels))
                        no_defects_in_seq = defect_count < self.min_defects_per_sequence
                        if no_defects_in_seq:
                            if total_sequences_nodefects_added >= total_sequences_with_defects or self.isOnlyDefective:
                                total_sequences_skipped += 1
                                continue

                        # format arrays
                        sequence_np = np.array(sequence, dtype=np.float32)              # [N, 320]
                        labels_np = np.array(seq_labels, dtype=np.float32)              # [N]
                        defects_np = np.array(
                            [[float(d[0]), float(d[1])] for d in seq_defects], dtype=np.float32
                        )                                                                  # [N,2] in [0,1]

                        # append original
                        self.signal_sets.append(sequence_np)
                        self.labels.append(labels_np)
                        self.defect_positions.append(defects_np)
                        if no_defects_in_seq:
                            total_sequences_nodefects_added += 1
                        else:
                            total_sequences_with_defects += 1

                        # augment: uniform pad lengths
                        if self.augment_uniform_pad_lengths:
                            for pad_len in self.augment_uniform_pad_lengths:
                                aug_seq, aug_def = self._augment_uniform(sequence_np, defects_np, pad_len)
                                self.signal_sets.append(aug_seq)
                                self.labels.append(labels_np.copy())
                                self.defect_positions.append(aug_def)

                        # augment: variable pad schedules
                        if self.augment_variable_pad_schedules:
                            for (pad_start, pad_end) in self.augment_variable_pad_schedules:
                                aug_seq, aug_def = self._augment_variable(sequence_np, defects_np, pad_start, pad_end)
                                self.signal_sets.append(aug_seq)
                                self.labels.append(labels_np.copy())
                                self.defect_positions.append(aug_def)

            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        print(f"Total beams: {total_beams}")
        print(f"Total sequences created: {total_sequences}")
        print(f"Sequences with defects (kept): {total_sequences_with_defects}")
        print(f"Sequences without defects (kept): {total_sequences_nodefects_added}")
        print(f"Sequences without defects (skipped): {total_sequences_skipped}")
        print(f"Defect sequence ratio: {total_sequences_with_defects / max(1,total_sequences):.3f}")

    # -------------------------
    # Augmentations
    # -------------------------
    def _augment_uniform(self, sequence: np.ndarray, defects: np.ndarray, pad_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pad each signal in the sequence with the SAME pad_len at the end, then resample to 320.
        Update defect positions by multiplying with scale = 320 / (320 + pad_len)."""
        N, L0 = sequence.shape
        Lp = L0 + max(0, int(pad_len))
        scale = L0 / float(Lp)  # = 320 / (320+pad)
        out_seq = np.empty_like(sequence)
        out_def = np.empty_like(defects)

        pad_vals = _make_pad_values(Lp - L0, mode=self.pad_mode, near_zero_range=self.near_zero_range)
        for i in range(N):
            padded = np.concatenate([sequence[i], pad_vals], axis=0)
            out_seq[i] = _resample_1d(padded, L0)
            s, e = _scale_positions_norm(float(defects[i,0]), float(defects[i,1]), scale)
            out_def[i, 0] = s
            out_def[i, 1] = e
        return out_seq, out_def

    def _augment_variable(self, sequence: np.ndarray, defects: np.ndarray, pad_start: int, pad_end: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pad signals with pad_len increasing linearly from pad_start to pad_end across the sequence, then resample to 320.
        Each signal's defect positions are scaled with its own scale = 320 / (320 + pad_len_i)."""
        N, L0 = sequence.shape
        pad_start = int(pad_start)
        pad_end = int(pad_end)
        if N <= 1:
            pads = [pad_end]
        else:
            pads = np.linspace(pad_start, pad_end, N, dtype=int).tolist()
        out_seq = np.empty_like(sequence)
        out_def = np.empty_like(defects)
        for i in range(N):
            pad_len_i = max(0, int(pads[i]))
            Lp = L0 + pad_len_i
            scale = L0 / float(Lp)
            pad_vals = _make_pad_values(Lp - L0, mode=self.pad_mode, near_zero_range=self.near_zero_range)
            padded = np.concatenate([sequence[i], pad_vals], axis=0)
            out_seq[i] = _resample_1d(padded, L0)
            s, e = _scale_positions_norm(float(defects[i,0]), float(defects[i,1]), scale)
            out_def[i, 0] = s
            out_def[i, 1] = e
        return out_seq, out_def

    # -------------------------
    # Torch Dataset API
    # -------------------------
    def __len__(self):
        return len(self.signal_sets)

    def __getitem__(self, idx):
        signals = torch.tensor(self.signal_sets[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        defect_positions = torch.tensor(self.defect_positions[idx], dtype=torch.float32)
        return signals, labels, defect_positions


# -----------------------------
# DataLoader factory
# -----------------------------

def get_defect_focused_dataloader(
    json_dir: str,
    batch_size: int = 4,
    seq_length: int = 50,
    shuffle: bool = True,
    num_workers: int = 4,
    validation_split: float = 0.2,
    min_defects_per_sequence: int = 1,
    isOnlyDefective: bool = False,
    augment_uniform_pad_lengths: Optional[List[int]] = None,                 # e.g., [50, 100, 320]
    augment_variable_pad_schedules: Optional[List[Tuple[int, int]]] = None,  # e.g., [(295, 320), (0, 320)]
    pad_mode: str = "zeros",
    near_zero_range: Tuple[float, float] = (0.0, 0.02),
):
    dataset = DefectFocusedJsonSignalDataset(
        json_dir=json_dir,
        seq_length=seq_length,
        min_defects_per_sequence=min_defects_per_sequence,
        isOnlyDefective=isOnlyDefective,
        augment_uniform_pad_lengths=augment_uniform_pad_lengths,
        augment_variable_pad_schedules=augment_variable_pad_schedules,
        pad_mode=pad_mode,
        near_zero_range=near_zero_range,
    )

    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Defect-focused dataset split: {train_size} training samples, {val_size} validation samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
