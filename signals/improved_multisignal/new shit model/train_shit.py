# train_hybridbinary_oldloader.py
import os, json, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --- your old loader ---
# expects either (signals, labels) or (signals, labels, meta)
from defect_focused_dataset import get_defect_focused_dataloader

# --- the model you pasted (save it as hybrid_binary_model.py) ---
from shit1 import HybridBinaryModel


def try_build_loaders(json_dir, batch_size, seq_length, val_split, num_workers, shuffle=True):
    """Be liberal about old signatures."""
    try:
        return get_defect_focused_dataloader(
            json_dir=json_dir,
            batch_size=batch_size,
            seq_length=seq_length,
            shuffle=shuffle,
            num_workers=num_workers,
            validation_split=val_split,
            min_defects_per_sequence=1,
        )
    except TypeError:
        try:
            return get_defect_focused_dataloader(
                root_path=json_dir,
                batch_size=batch_size,
                seq_length=seq_length,
                shuffle=shuffle,
                num_workers=num_workers,
                validation_split=val_split,
                min_defects_per_sequence=1,
            )
        except TypeError:
            return get_defect_focused_dataloader(
                json_dir=json_dir,
                batch_size=batch_size,
                seq_length=seq_length,
                shuffle=shuffle,
                validation_split=val_split,
            )


def to_float(x):
    return float(x.detach().item() if isinstance(x, torch.Tensor) else x)


def main():
    ap = argparse.ArgumentParser("Train HybridBinaryModel on OLD loader (detection only)")
    # paths
    cwd = Path.cwd()
    ap.add_argument("--json_dir", type=str, default=str(cwd.parent / "json_data_0717"))
    ap.add_argument("--save_dir", type=str, default="models/shit1")
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
    # report
    ap.add_argument("--thr", type=float, default=0.5, help="threshold for PR/F1 reporting")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_loader, val_loader = try_build_loaders(
        json_dir=args.json_dir,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        val_split=args.val_split,
        num_workers=args.num_workers,
        shuffle=True,
    )

    # model
    model = HybridBinaryModel(signal_length=320).to(device)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3)

    history = {
        "epoch": [],
        "train_loss": [], "val_loss": [],
        "train_prec": [], "train_rec": [], "train_f1": [],
        "val_prec": [], "val_rec": [], "val_f1": [],
        "lr": [],
    }

    best_val = float("inf")

    def step_loop(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total_loss, TP, FP, FN = 0.0, 0, 0, 0
        with torch.set_grad_enabled(train):
            for batch in tqdm(loader, desc=("Train" if train else "Val")):
                if isinstance(batch, (list, tuple)):
                    signals, labels = batch[:2]
                else:
                    signals, labels = batch["signals"], batch["labels"]
                signals = signals.to(device).float()  # [B,N,S]
                labels = labels.to(device).float()    # [B,N] with {0,1}

                if train:
                    optimizer.zero_grad()

                # model returns probabilities already
                probs = model(signals)               # [B,N] in [0,1]
                loss = F.binary_cross_entropy(probs, labels)

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()

                total_loss += to_float(loss)

                # simple metrics @ threshold
                preds = (probs >= args.thr).long()
                lbls = labels.long()
                TP += int(((preds == 1) & (lbls == 1)).sum().item())
                FP += int(((preds == 1) & (lbls == 0)).sum().item())
                FN += int(((preds == 0) & (lbls == 1)).sum().item())

        n = max(1, len(loader))
        loss_avg = total_loss / n
        prec = TP / max(1, TP + FP)
        rec  = TP / max(1, TP + FN)
        f1   = 2 * prec * rec / max(1e-8, (prec + rec)) if (prec + rec) > 0 else 0.0
        return loss_avg, prec, rec, f1

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_p, tr_r, tr_f1 = step_loop(train_loader, train=True)
        va_loss, va_p, va_r, va_f1 = step_loop(val_loader, train=False)

        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}: tr {tr_loss:.4f} (P/R/F1 {tr_p:.3f}/{tr_r:.3f}/{tr_f1:.3f}) | "
            f"va {va_loss:.4f} (P/R/F1 {va_p:.3f}/{va_r:.3f}/{va_f1:.3f}) | lr {lr_now:.3e}"
        )

        # save
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": va_loss,
            "cfg": {
                "signal_length": 320,
                "threshold": args.thr,
            },
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"epoch_{epoch:02d}.pth"))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))

        # log
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_prec"].append(tr_p);    history["val_prec"].append(va_p)
        history["train_rec"].append(tr_r);     history["val_rec"].append(va_r)
        history["train_f1"].append(tr_f1);     history["val_f1"].append(va_f1)
        history["lr"].append(lr_now)

        with open(os.path.join(args.save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
