# train.py
import os, json, argparse, math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# from defect_focused_dataset import get_defect_focused_dataloader
from defect_focused_dataset_aug import get_defect_focused_dataloader

from hybrid1d_detloc import (
    Hybrid1D_DetLoc,
    DetLocCriterion,
    DetLocLossCfg,
    TrainCfg,
)
from pathlib import Path

current_directory = Path.cwd()
DS_DIR = current_directory.parent / "json_data_0717"



def _extract_centers_widths(labels, defect_positions):
    """
    labels: [B,N] in {0,1}
    defect_positions: [B,N,2] with [start,end] in [0,1] (0,0 if none)
    returns centers01, widths01 (both [B,N], float) with arbitrary values for negatives
    """
    # centers=(s+e)/2, width=(e-s); clamp to [0,1]
    starts = defect_positions[..., 0]
    ends   = defect_positions[..., 1]
    centers = ((starts + ends) * 0.5).clamp(0.0, 1.0)
    widths  = (ends - starts).clamp(1e-6, 1.0)
    # negatives will be ignored by the loss via pos_mask, so values don't matter
    return centers, widths

def plot_history(hist):
    epochs = hist["epoch"]
    plt.figure(figsize=(12,8))

    # Losses
    plt.subplot(2,2,1)
    plt.plot(epochs, hist["train_loss"], label="train")
    plt.plot(epochs, hist["val_loss"],   label="val")
    plt.title("Total loss"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True); plt.legend()

    plt.subplot(2,2,2)
    plt.plot(epochs, hist["train_obj"], label="train_obj")
    plt.plot(epochs, hist["val_obj"],   label="val_obj")
    plt.title("Objectness loss"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True); plt.legend()

    plt.subplot(2,2,3)
    plt.plot(epochs, hist["train_l1"], label="train_l1")
    plt.plot(epochs, hist["val_l1"],   label="val_l1")
    plt.title("L1(center,width)"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True); plt.legend()

    plt.subplot(2,2,4)
    plt.plot(epochs, hist["train_iou"], label="train_iou")
    plt.plot(epochs, hist["val_iou"],   label="val_iou")
    plt.title("1 - IoU1D"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()  # display only (no saving)

def _to_float(x):
    # works for Tensor or float
    return float(x.detach().item() if isinstance(x, torch.Tensor) else x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", default=DS_DIR, help="Folder with JSON signals")
    ap.add_argument("--save_dir", default="models/h1d_detloc_run")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_length", type=int, default=50)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=1.5e-2)
    ap.add_argument("--focal", action="store_true", help="Use focal BCE for objectness")
    ap.add_argument("--no_focal", dest="focal", action="store_false")
    ap.set_defaults(focal=True)
    # optional: augmentation params if using defect_focused_dataset_aug
    ap.add_argument("--uniform_pads", type=int, nargs="*", default=[160, 320, 640], help="e.g. 160 320 640")
    ap.add_argument("--var_pads", type=int, nargs="*", default=[295, 320, 430, 480], help="pairs: e.g. 295 320  0 320")
    ap.add_argument("--pad_mode", choices=["zeros","near_zero"], default="zeros")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Build variable pad schedules list of tuples
    if len(args.var_pads) % 2 != 0:
        raise SystemExit("var_pads must be even length: pairs of (start end)")
    var_sched = [(args.var_pads[i], args.var_pads[i+1]) for i in range(0, len(args.var_pads), 2)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_defect_focused_dataloader(
        json_dir=args.json_dir,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        shuffle=True,
        num_workers=4,
        validation_split=args.val_split,
        min_defects_per_sequence=1,
        isOnlyDefective=False,
        augment_uniform_pad_lengths=args.uniform_pads,
        augment_variable_pad_schedules=var_sched,
        pad_mode=args.pad_mode,
    )

    model = Hybrid1D_DetLoc(signal_length=320).to(device)

    loss_cfg = DetLocLossCfg(focal=args.focal)
    opt_cfg  = TrainCfg(lr=args.lr, weight_decay=args.wd, epochs=args.epochs, clip_grad=1.0)
    criterion = DetLocCriterion(loss_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3)

    hist = {
        "epoch": [],
        "train_loss": [], "train_obj": [], "train_l1": [], "train_iou": [],
        "val_loss":   [], "val_obj":   [], "val_l1":   [], "val_iou":   [],
        "lr": [],
    }

    best_val = float("inf")

    for epoch in range(1, opt_cfg.epochs + 1):
        # ----------------- train
        model.train()
        acc = {"loss":0.0, "obj":0.0, "l1":0.0, "iou":0.0}
        for batch in tqdm(train_loader, desc=f"Train {epoch}"):
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    signals, labels, defect_pos = batch
                    meta = {"centers": None, "widths": None}
                else:
                    signals, labels = batch
                    defect_pos = None
                    meta = {"centers": None, "widths": None}
            else:
                # dict format not expected for your current loader
                signals = batch["signals"]; labels = batch["labels"]; defect_pos = batch.get("defect_positions", None)
                meta = {}

            signals = signals.to(device).float()     # [B,N,S]
            labels  = labels.to(device).float()      # [B,N]
            centers = widths = None
            if defect_pos is not None:
                defect_pos = defect_pos.to(device).float()  # [B,N,2] (start,end) in [0,1]
                centers, widths = _extract_centers_widths(labels, defect_pos)

            out = model(signals)
            losses = criterion(out, labels, centers, widths)

            optimizer.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.clip_grad)
            optimizer.step()

            acc["loss"] += _to_float(losses["loss"])
            acc["obj"] += _to_float(losses["loss_obj"])
            acc["l1"] += _to_float(losses["loss_l1"])
            acc["iou"] += _to_float(losses["loss_iou"])

        ntr = max(1, len(train_loader))
        tr_loss = acc["loss"]/ntr; tr_obj = acc["obj"]/ntr; tr_l1 = acc["l1"]/ntr; tr_iou = acc["iou"]/ntr

        # ----------------- val
        model.eval()
        acc = {"loss":0.0, "obj":0.0, "l1":0.0, "iou":0.0}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        signals, labels, defect_pos = batch
                    else:
                        signals, labels = batch
                        defect_pos = None
                else:
                    signals = batch["signals"]; labels = batch["labels"]; defect_pos = batch.get("defect_positions", None)

                signals = signals.to(device).float()
                labels  = labels.to(device).float()
                centers = widths = None
                if defect_pos is not None:
                    defect_pos = defect_pos.to(device).float()
                    centers, widths = _extract_centers_widths(labels, defect_pos)

                out = model(signals)
                losses = criterion(out, labels, centers, widths)

                acc["loss"] += float(losses["loss"].item())
                acc["obj"]  += float(losses["loss_obj"].item())
                acc["l1"]   += float(losses["loss_l1"].item())
                acc["iou"]  += float(losses["loss_iou"].item())

        nva = max(1, len(val_loader))
        va_loss = acc["loss"]/nva; va_obj = acc["obj"]/nva; va_l1 = acc["l1"]/nva; va_iou = acc["iou"]/nva

        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}: tr {tr_loss:.4f}/{tr_obj:.4f}/{tr_l1:.4f}/{tr_iou:.4f} | "
              f"va {va_loss:.4f}/{va_obj:.4f}/{va_l1:.4f}/{va_iou:.4f} | lr {lr_now:.3e}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": va_loss,
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"epoch_{epoch:02d}.pth"))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))

        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr_loss); hist["train_obj"].append(tr_obj); hist["train_l1"].append(tr_l1); hist["train_iou"].append(tr_iou)
        hist["val_loss"].append(va_loss);   hist["val_obj"].append(va_obj);   hist["val_l1"].append(va_l1);   hist["val_iou"].append(va_iou)
        hist["lr"].append(lr_now)

        with open(os.path.join(args.save_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)

    plot_history(hist)

if __name__ == "__main__":
    # avoid worker re-import noise or CUDA dataloader hangs by setting
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
