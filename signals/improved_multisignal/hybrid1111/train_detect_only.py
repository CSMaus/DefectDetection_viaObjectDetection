# train_detect_only.py
import os, json, argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from defect_focused_dataset import get_defect_focused_dataloader

from hybrid1d_detloc import (
    Hybrid1D_DetLoc,
    DetLocCriterion,
    DetLocLossCfg,
)

# ------------- utils -------------
def to_float(x):
    return float(x.detach().item() if isinstance(x, torch.Tensor) else x)

def try_build_loaders(json_dir, batch_size, seq_length, val_split, num_workers, shuffle=True):
    """
    Old loaders in your repo had slightly different signatures across versions.
    This tries a few common ones so you don't have to edit the script.
    """
    # Attempt 1: json_dir + common args
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

    # Attempt 2: root_path instead of json_dir
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

    # Attempt 3: minimal signature
    return get_defect_focused_dataloader(
        json_dir=json_dir,
        batch_size=batch_size,
        seq_length=seq_length,
        shuffle=shuffle,
        validation_split=val_split
    )

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser("Detection-only training (old loader)")
    # paths
    cwd = Path.cwd()
    default_ds = cwd.parent / "json_data_0717"
    ap.add_argument("--json_dir", type=str, default=str(default_ds), help="Folder with JSON signals")
    ap.add_argument("--save_dir", type=str, default="models/h1d_detect_only")
    # data
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_length", type=int, default=50)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=4)
    # train
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=1.5e-2)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    # loss
    ap.add_argument("--focal", action="store_true", help="Use focal BCE for objectness")
    ap.add_argument("--no_focal", dest="focal", action="store_false")
    ap.set_defaults(focal=True)
    # which scales to use for obj loss
    ap.add_argument("--scales", type=str, default="P3,P4,P5",
                    help="Comma-separated subset of {P3,P4,P5} to use for obj loss, e.g. P3 or P3,P4")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders (old dataloader; detection labels only)
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

    # loss: we’ll use DetLocCriterion with centers=None/widths=None => detection-only.
    # We will ZERO-OUT (ignore) contributions from scales you don’t want.
    loss_cfg = DetLocLossCfg(focal=args.focal, lambda_l1=0.0, lambda_iou=0.0)
    base_criterion = DetLocCriterion(loss_cfg).to(device)

    # Wrap criterion to drop unwanted scales
    wanted = set([s.strip() for s in args.scales.split(",") if s.strip() in {"P3","P4","P5"}])
    if not wanted:
        wanted = {"P3"}  # safe default

    class ObjOnlyCriterion(nn.Module):
        def __init__(self, base, wanted_scales):
            super().__init__()
            self.base = base
            self.wanted = wanted_scales
        def forward(self, out, labels):
            # Keep only selected scales by zeroing out others before feeding to base
            filtered = {}
            for k, v in out.items():
                if k.startswith("obj_logits_"):
                    scale = k.split("_")[-1]  # P3/P4/P5
                    if scale in self.wanted:
                        filtered[k] = v
                    else:
                        # zero out contribution
                        filtered[k] = torch.zeros_like(v)
                elif k.startswith("reg_"):
                    # not used; keep shape to satisfy base but set zeros
                    filtered[k] = torch.zeros_like(v)
            # centers/widths=None -> detection only in base criterion
            return self.base(filtered, labels, centers01=None, widths01=None)

    criterion = ObjOnlyCriterion(base_criterion, wanted)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3)

    history = {
        "epoch": [],
        "train_loss": [], "train_obj": [],
        "val_loss": [],   "val_obj":  [],
        "lr": [],
        "scales": sorted(list(wanted)),
    }

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # -------- train
        model.train()
        acc_loss = 0.0
        acc_obj  = 0.0
        for batch in tqdm(train_loader, desc=f"Train {epoch}"):
            # old loader should return (signals, labels)
            if isinstance(batch, (list, tuple)):
                signals, labels = batch[:2]
            else:  # dict-like fallback
                signals, labels = batch["signals"], batch["labels"]

            signals = signals.to(device).float()   # [B,N,S]
            labels  = labels.to(device).float()    # [B,N]

            out = model(signals)
            losses = criterion(out, labels)

            optimizer.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            acc_loss += to_float(losses["loss"])
            acc_obj  += to_float(losses["loss_obj"])
        ntr = max(1, len(train_loader))
        tr_loss = acc_loss / ntr
        tr_obj  = acc_obj  / ntr

        # -------- val
        model.eval()
        acc_loss = 0.0
        acc_obj  = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                if isinstance(batch, (list, tuple)):
                    signals, labels = batch[:2]
                else:
                    signals, labels = batch["signals"], batch["labels"]
                signals = signals.to(device).float()
                labels  = labels.to(device).float()
                out = model(signals)
                losses = criterion(out, labels)
                acc_loss += to_float(losses["loss"])
                acc_obj  += to_float(losses["loss_obj"])
        nva = max(1, len(val_loader))
        va_loss = acc_loss / nva
        va_obj  = acc_obj  / nva

        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}: tr {tr_loss:.4f}/{tr_obj:.4f} | va {va_loss:.4f}/{va_obj:.4f} | lr {lr_now:.3e}")

        # save
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": va_loss,
            "cfg": {
                "scales": sorted(list(wanted)),
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
        history["train_loss"].append(tr_loss)
        history["train_obj"].append(tr_obj)
        history["val_loss"].append(va_loss)
        history["val_obj"].append(va_obj)
        history["lr"].append(lr_now)

        with open(os.path.join(args.save_dir, "history_detect_only.json"), "w") as f:
            json.dump(history, f, indent=2)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
