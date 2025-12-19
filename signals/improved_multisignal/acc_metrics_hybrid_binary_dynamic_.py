import os
import json
import math
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from detection_models.hybrid_binary_dynamic import HybridBinaryModel
from detection_models.hybrid_binary import HybridBinaryModel

try:
    from defect_focused_dataset import get_defect_focused_dataloader as get_loader
    LOADER_NAME = "defect_focused_dataset"
except Exception:
    from defect_focused_dataset_aug import get_defect_focused_dataloader as get_loader
    LOADER_NAME = "defect_focused_dataset_aug"


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return safe_div(num, math.sqrt(den)) if den else 0.0


def load_legacy_mha_checkpoint_into_tiny(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    old_sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    new_sd = model.state_dict()

    for k in new_sd.keys():
        if ".self_attn." in k:
            continue
        if k in old_sd and old_sd[k].shape == new_sd[k].shape:
            new_sd[k] = old_sd[k]

    for i in range(len(model.transformer_layers)):
        prefix_old = f"transformer_layers.{i}.self_attn"
        prefix_q = f"transformer_layers.{i}.self_attn.q"
        prefix_k = f"transformer_layers.{i}.self_attn.k"
        prefix_v = f"transformer_layers.{i}.self_attn.v"
        prefix_o = f"transformer_layers.{i}.self_attn.o"

        if (prefix_old + ".in_proj_weight") not in old_sd:
            continue

        Wqkv = old_sd[prefix_old + ".in_proj_weight"]
        bqkv = old_sd[prefix_old + ".in_proj_bias"]
        Wout = old_sd[prefix_old + ".out_proj.weight"]
        bout = old_sd[prefix_old + ".out_proj.bias"]

        D = Wout.shape[0]
        new_sd[prefix_q + ".weight"] = Wqkv[0:D, :].clone()
        new_sd[prefix_q + ".bias"] = bqkv[0:D].clone()
        new_sd[prefix_k + ".weight"] = Wqkv[D:2 * D, :].clone()
        new_sd[prefix_k + ".bias"] = bqkv[D:2 * D].clone()
        new_sd[prefix_v + ".weight"] = Wqkv[2 * D:3 * D, :].clone()
        new_sd[prefix_v + ".bias"] = bqkv[2 * D:3 * D].clone()
        new_sd[prefix_o + ".weight"] = Wout.clone()
        new_sd[prefix_o + ".bias"] = bout.clone()

    model.load_state_dict(new_sd, strict=False)
    return model


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    tp = fp = tn = fn = 0
    total = 0

    for signals, labels, _meta in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        probs = model(signals)
        preds = (probs >= threshold).float()

        p = preds.reshape(-1)
        y = (labels > 0.5).float().reshape(-1)

        tp += int(((p == 1) & (y == 1)).sum().item())
        fp += int(((p == 1) & (y == 0)).sum().item())
        tn += int(((p == 0) & (y == 0)).sum().item())
        fn += int(((p == 0) & (y == 1)).sum().item())
        total += y.numel()

    acc = safe_div(tp + tn, total)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    f1 = safe_div(2 * prec * rec, prec + rec)
    bal_acc = 0.5 * (rec + spec)
    mcc_val = mcc(tp, tn, fp, fn)

    counts = {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Total signals": total,
        "Defect signals (P=TP+FN)": tp + fn,
        "No-defect signals (N=TN+FP)": tn + fp,
    }

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Specificity": spec,
        "Balanced Accuracy": bal_acc,
        "MCC": mcc_val,
        "Threshold": threshold,
    }

    return counts, metrics


def save_table_png(df, out_path, title=None, float_fmt="{:.6f}"):
    fig, ax = plt.subplots(figsize=(10, 2.2 + 0.25 * len(df)))
    ax.axis("off")
    df_disp = df.copy()
    for c in df_disp.columns:
        if pd.api.types.is_float_dtype(df_disp[c]):
            df_disp[c] = df_disp[c].map(lambda x: float_fmt.format(x))
    tbl = ax.table(cellText=df_disp.values, colLabels=df_disp.columns, rowLabels=df_disp.index,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    if title:
        ax.set_title(title, pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_like_yours(tp, fp, fn, tn, out_counts_path, out_pct_path):
    cm_counts = np.array([[tp, fp],
                          [fn, tn]], dtype=np.int64)

    r1 = tp + fp
    r2 = fn + tn
    cm_pct = np.array([[safe_div(tp, r1) * 100.0, safe_div(fp, r1) * 100.0],
                       [safe_div(fn, r2) * 100.0, safe_div(tn, r2) * 100.0]], dtype=np.float64)

    def draw(mat, fmt, title, out_path):
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Defect", "No defect"])
        ax.set_yticklabels(["Defect", "No defect"])
        ax.set_xlabel("True defects")
        ax.set_ylabel("Predicted defects")
        ax.set_title(title)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=12)
        plt.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    draw(cm_counts, "{}", "Confusion Matrix", out_counts_path)
    draw(cm_pct, "{:.2f}%", "Confusion Matrix (%)", out_pct_path)


def main():
    # "json_eval_original_15" # "json_data-test"
    data_dir = "json_eval_original_15"
    ckpt_path = r"models\HybridBinaryModel_20251118_1751\best_detection.pth"
    out_dir = "eval_results_original"

    batch_size = 8
    seq_length = 50
    signal_length = 320
    threshold = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    loader_kwargs = dict(
        batch_size=batch_size,
        seq_length=seq_length,
        shuffle=False,
        validation_split=0.0,
        min_defects_per_sequence=0,
        isOnlyDefective=False,
    )
    if "aug" in LOADER_NAME:
        loader_kwargs["augment_uniform_pad_lengths"] = []

    loader_out = get_loader(data_dir, **loader_kwargs)
    test_loader = loader_out[0] if isinstance(loader_out, tuple) else loader_out

    model = HybridBinaryModel(
        signal_length=signal_length,
        hidden_sizes=[256, 192, 64],
        num_heads=8,
        dropout=0.15,
        num_transformer_layers=4
    ).to(device)

    model = load_legacy_mha_checkpoint_into_tiny(model, ckpt_path, device)
    # ckpt = torch.load(ckpt_path, map_location=device)
    # state = ckpt.get("model_state_dict", ckpt)
    # model.load_state_dict(state, strict=True)

    counts, metrics = evaluate(model, test_loader, device, threshold=threshold)

    df_counts = pd.DataFrame([counts]).T
    df_counts.columns = ["Count"]

    df_metrics = pd.DataFrame([metrics]).T
    df_metrics.columns = ["Value"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "timestamp": stamp,
        "device": str(device),
        "loader": LOADER_NAME,
        "data_dir": os.path.abspath(data_dir),
        "checkpoint": os.path.abspath(ckpt_path),
        "seq_length": seq_length,
        "signal_length": signal_length,
        "threshold": threshold,
    }
    with open(os.path.join(out_dir, f"eval_meta_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    df_counts.to_csv(os.path.join(out_dir, f"table_counts_{stamp}.csv"))
    df_metrics.to_csv(os.path.join(out_dir, f"table_metrics_{stamp}.csv"))

    save_table_png(df_counts, os.path.join(out_dir, f"table_counts_{stamp}.png"), "Table A — Signal Counts")
    save_table_png(df_metrics, os.path.join(out_dir, f"table_metrics_{stamp}.png"), "Table B — Metrics")

    save_confusion_like_yours(
        tp=counts["TP"], fp=counts["FP"], fn=counts["FN"], tn=counts["TN"],
        out_counts_path=os.path.join(out_dir, f"confusion_counts_{stamp}.png"),
        out_pct_path=os.path.join(out_dir, f"confusion_percent_{stamp}.png"),
    )

    print(df_counts)
    print(df_metrics)


if __name__ == "__main__":
    main()
