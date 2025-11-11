# train_detloc_from_oldloader.py
import os, json, argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from defect_focused_dataset import get_defect_focused_dataloader  # your old loader
from hybrid1d_detloc import (
    Hybrid1D_DetLoc,
    DetLocCriterion,
    DetLocLossCfg,
)

# ---------------- utils ----------------
def to_float(x):
    return float(x.detach().item() if isinstance(x, torch.Tensor) else x)

def _extract_centers_widths(labels, defect_positions):
    """
    labels: [B,N] in {0,1}
    defect_positions: [B,N,2] with [start,end] in [0,1] (0,0 if none)
    returns centers01, widths01 (both [B,N], float); negatives ignored by loss via pos_mask
    """
    starts = defect_positions[..., 0]
    ends   = defect_positions[..., 1]
    centers = ((starts + ends) * 0.5).clamp(0.0, 1.0)
    widths  = (ends - starts).clamp(1e-6, 1.0)
    return centers, widths

def try_build_loaders(json_dir, batch_size, seq_length, val_split, num_workers, shuffle=True):
    """Be tolerant to signature differences across your old loader variants."""
    # Attempt 1
    try:
        return get_defect_focused_dataloader(
            json_dir=json_dir,
            batch_size=batch_size,
            seq_length=seq_length,
            shuffle=shuffle,
            num_workers=num_workers,
            validation_split=val_split,
            min_defects_per_sequence=1
        )
    except TypeError:
        pass
    # Attempt 2
    try:
        return get_defect_focused_dataloader(
            root_path=json_dir,
            batch_size=batch_size,
            seq_length=seq_length,
            shuffle=shuffle,
            num_workers=num_workers,
            validation_split=val_split,
            min_defects_per_sequence=1
        )
    except TypeError:
        pass
    # Fallback (minimal)
    return get_defect_focused_dataloader(
        json_dir=json_dir,
        batch_size=batch_size,
        seq_length=seq_length,
        shuffle=shuffle,
        validation_split=val_split
    )

# --------------- main ------------------
def main():
    ap = argparse.ArgumentParser("Det+Loc training using OLD loader (auto-detects position targets)")
    # paths
    cwd = Path.cwd()
    default_ds = cwd.parent / "json_data_0717"
    ap.add_argument("--json_dir", type=str, default=str(default_ds), help="Folder with JSON signals")
    ap.add_argument("--save_dir", type=str, default="models/h1d_detloc_from_oldloader")
    # data
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_length", type=int, default=50)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=4)
    # train
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=1.5e-2)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    # loss
    ap.add_argument("--focal", action="store_true", help="Use focal BCE for objectness")
    ap.add_argument("--no_focal", dest="focal", action="store_false")
    ap.set_defaults(focal=True)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders (old dataloader)
    train_loader, val_loader = try_build_loaders(
        json_dir=args.json_dir,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        val_split=args.val_split,
        num_workers=args.num_workers,
        shuffle=True
    )

    # model
    model = Hybrid1D_DetLoc(signal_length=320).to(device)

    # criterion (supports det-only or det+loc depending on provided targets)
    loss_cfg = DetLocLossCfg(focal=args.focal)
    criterion = DetLocCriterion(loss_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3)

    history = {
        "epoch": [],
        "train_loss": [], "train_obj": [], "train_l1": [], "train_iou": [],
        "val_loss":   [], "val_obj":   [], "val_l1":   [], "val_iou":   [],
        "lr": [],
    }

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---------- train
        model.train()
        acc = {"loss":0.0, "obj":0.0, "l1":0.0, "iou":0.0}
        for batch in tqdm(train_loader, desc=f"Train {epoch}"):
            # Accept common batch shapes from your old loader
            defect_pos = None
            if isinstance(batch, (list, tuple)):
                # (signals, labels) or (signals, labels, defect_positions)
                if len(batch) >= 2:
                    signals, labels = batch[:2]
                if len(batch) >= 3:
                    defect_pos = batch[2]
            else:
                # dict-like
                signals = batch["signals"]; labels = batch["labels"]
                defect_pos = batch.get("defect_positions", None)

            signals = signals.to(device).float()   # [B,N,S]
            labels  = labels.to(device).float()    # [B,N]

            centers = widths = None
            if defect_pos is not None:
                defect_pos = defect_pos.to(device).float()  # [B,N,2] normalized
                centers, widths = _extract_centers_widths(labels, defect_pos)

            out = model(signals)
            losses = criterion(out, labels, centers, widths)

            optimizer.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            acc["loss"] += to_float(losses["loss"])
            acc["obj"]  += to_float(losses["loss_obj"])
            acc["l1"]   += to_float(losses["loss_l1"])
            acc["iou"]  += to_float(losses["loss_iou"])

        ntr = max(1, len(train_loader))
        tr_loss = acc["loss"]/ntr; tr_obj = acc["obj"]/ntr; tr_l1 = acc["l1"]/ntr; tr_iou = acc["iou"]/ntr

        # ---------- val
        model.eval()
        acc = {"loss":0.0, "obj":0.0, "l1":0.0, "iou":0.0}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                defect_pos = None
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        signals, labels = batch[:2]
                    if len(batch) >= 3:
                        defect_pos = batch[2]
                else:
                    signals = batch["signals"]; labels = batch["labels"]
                    defect_pos = batch.get("defect_positions", None)

                signals = signals.to(device).float()
                labels  = labels.to(device).float()

                centers = widths = None
                if defect_pos is not None:
                    defect_pos = defect_pos.to(device).float()
                    centers, widths = _extract_centers_widths(labels, defect_pos)

                out = model(signals)
                losses = criterion(out, labels, centers, widths)

                acc["loss"] += to_float(losses["loss"])
                acc["obj"]  += to_float(losses["loss_obj"])
                acc["l1"]   += to_float(losses["loss_l1"])
                acc["iou"]  += to_float(losses["loss_iou"])

        nva = max(1, len(val_loader))
        va_loss = acc["loss"]/nva; va_obj = acc["obj"]/nva; va_l1 = acc["l1"]/nva; va_iou = acc["iou"]/nva

        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}: tr {tr_loss:.4f}/{tr_obj:.4f}/{tr_l1:.4f}/{tr_iou:.4f} | "
              f"va {va_loss:.4f}/{va_obj:.4f}/{va_l1:.4f}/{va_iou:.4f} | lr {lr_now:.3e}")

        # save
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": va_loss,
            "cfg": {
                "signal_length": 320,
                "focal": args.focal,
            },
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"epoch_{epoch:02d}.pth"))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))

        # log
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss); history["train_obj"].append(tr_obj)
        history["train_l1"].append(tr_l1);     history["train_iou"].append(tr_iou)
        history["val_loss"].append(va_loss);   history["val_obj"].append(va_obj)
        history["val_l1"].append(va_l1);       history["val_iou"].append(va_iou)
        history["lr"].append(lr_now)

        with open(os.path.join(args.save_dir, "history_detloc.json"), "w") as f:
            json.dump(history, f, indent=2)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
