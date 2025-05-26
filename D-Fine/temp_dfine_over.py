import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DFineForObjectDetection,
    AutoImageProcessor,
    DFineConfig,
)
from transformers.models.d_fine.modeling_d_fine import (
    weighting_function,
    distance2bbox,
)

# ────────────────────────────────────────────────────────────────────────────────
# 1) Sequence‐of‐50 Dataset (unchanged)
# ────────────────────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(
        self,
        ds_root: str,
        ann_file: str,
        seq_len: int,
        processor=None,
        label_map=None,
    ):
        with open(ann_file, "r") as f:
            self.anns = json.load(f)

        # build or reuse label_map
        if label_map is None:
            all_labels = {
                obj["label"]
                for folder in self.anns.values()
                for annots in folder.values()
                for obj in annots
            }
            self.label_map = {lab: i for i, lab in enumerate(sorted(all_labels))}
        else:
            self.label_map = label_map

        self.seq_len = seq_len
        self.processor = processor

        # collect sequences of up to seq_len frames
        self.sequences = []
        for folder in sorted(os.listdir(ds_root)):
            fp = os.path.join(ds_root, folder)
            if not os.path.isdir(fp):
                continue
            imgs = sorted(
                [os.path.join(fp, fn) for fn in os.listdir(fp) if fn.endswith(".png")],
                key=lambda x: int(os.path.basename(x).split(".")[0]),
            )
            N = len(imgs)
            if N <= seq_len:
                self.sequences.append(imgs)
            else:
                for start in range(0, N, seq_len):
                    end = start + seq_len
                    if end <= N:
                        self.sequences.append(imgs[start:end])
                if N % seq_len:
                    self.sequences.append(imgs[-seq_len:])

        # keep only sequences with ≥1 GT
        filtered = []
        for seq in self.sequences:
            fld = os.path.basename(os.path.dirname(seq[0]))
            if any(self.anns[fld].get(os.path.basename(p), []) for p in seq):
                filtered.append(seq)
        self.sequences = filtered

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        paths = self.sequences[idx]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        proc = self.processor(images=imgs, return_tensors="pt")
        pv = proc["pixel_values"]  # (T,3,H,W)

        sizes, targets = [], []
        for p, im in zip(paths, imgs):
            sizes.append(im.size[::-1])
            fld = os.path.basename(os.path.dirname(p))
            annots = self.anns[fld].get(os.path.basename(p), [])
            boxes, labels = [], []
            W, H = im.size
            for o in annots:
                x1, x2, y1, y2 = o["bbox"]
                x, y = min(x1, x2), min(y1, y2)
                w, h = abs(x2 - x1), abs(y2 - y1)
                boxes.append([(x + w/2)/W, (y + h/2)/H, w/W, h/H])
                labels.append(self.label_map[o["label"]])

            if boxes:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            targets.append({"boxes": boxes, "class_labels": labels})

        return {"pixel_values": pv, "sizes": sizes, "targets": targets}


# ────────────────────────────────────────────────────────────────────────────────
# 2) TemporalDFine with defect‐only head (uses ignore_mismatched_sizes)
# ────────────────────────────────────────────────────────────────────────────────
class TemporalDFine(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        num_defect_classes: int,
        id2label: dict,
    ):
        super().__init__()

        # 1) custom config: your_num_labels = defects + 1 no-object
        cfg = DFineConfig.from_pretrained(
            checkpoint,
            num_labels=num_defect_classes + 1,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
        )

        # 2) load model & processor, skipping old classifier weights
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
        self.dfine = DFineForObjectDetection.from_pretrained(
            checkpoint,
            config=cfg,
            ignore_mismatched_sizes=True,
        )

        # 3) freeze everything
        for p in self.dfine.parameters():
            p.requires_grad = False

        # 4) temporal encoder
        d_model = self.dfine.config.hidden_size
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=4)
        for m in self.temporal_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 5) unfreeze your new classifier
        for p in self.dfine.class_embed[-1].parameters():
            p.requires_grad = True

        # 6) carry over bbox head & integral logic
        self.bbox_head = self.dfine.bbox_embed[-1]
        self.integral = self.dfine.model.decoder.integral
        self.up = self.dfine.model.decoder.up
        self.reg_scale = self.dfine.config.reg_scale
        self.max_num_bins = self.dfine.config.max_num_bins

    def forward(self, pixel_values, sizes, targets=None):
        out = self.dfine.model(pixel_values=pixel_values)
        feats = out.last_hidden_state
        init_ref = out.init_reference_points

        fused = self.temporal_encoder(feats)
        cls_logits = self.dfine.class_embed[-1](fused)
        bbox_dist = self.bbox_head(fused)

        # Wn = weighting_function(self.max_num_bins, self.up, self.reg_scale)
        # distances = self.integral(bbox_dist, Wn)
        # bbox_pred = distance2bbox(init_ref, distances, self.reg_scale).clamp(0, 1)
        # 4) turn distributions → normalized [0,1] boxes
        # manually build the “projection” vector Wₙ instead of using weighting_function():
        Wn = torch.arange(
            self.max_num_bins + 1,
            device=bbox_dist.device,
            dtype=bbox_dist.dtype,
        ) * self.up / self.reg_scale  # shape: (bins+1,)

        distances = self.integral(bbox_dist, Wn)  # (T, Q, 4)
        bbox_pred = distance2bbox(init_ref, distances, self.reg_scale).clamp(0, 1)

        # inference
        if targets is None:
            results = []
            for i, sz in enumerate(sizes):
                res = self.processor.post_process_object_detection(
                    {"logits": cls_logits[i : i + 1], "pred_boxes": bbox_pred[i : i + 1]},
                    target_sizes=torch.tensor([sz]),
                    threshold=0.3,
                )[0]
                results.append(res)
            return results

        # training loss
        losses = []
        noobj = cls_logits.size(-1) - 1
        for i, gt in enumerate(targets):
            pl = cls_logits[i : i + 1]
            pb = bbox_pred[i : i + 1]
            if gt["boxes"].numel() > 0:
                lc, _, _ = self.dfine.loss_function(
                    logits=pl,
                    labels=[gt],
                    device=pl.device,
                    pred_boxes=pb,
                    config=self.dfine.config,
                    outputs_class=pl.unsqueeze(0),
                    outputs_coord=pb.unsqueeze(0),
                    denoising_meta_values=None,
                    predicted_corners=None,
                    initial_reference_points=init_ref[i : i + 1].unsqueeze(0),
                )
                losses.append(lc)
            else:
                logp = pl.squeeze(0)
                tgt = torch.full((logp.size(0),), noobj, dtype=torch.long, device=pl.device)
                losses.append(F.cross_entropy(logp, tgt))

        return torch.stack(losses).mean(), {}


# ────────────────────────────────────────────────────────────────────────────────
# 3) collate + training loop (with both batch & epoch loss saved)
# ────────────────────────────────────────────────────────────────────────────────
def identity_collate(batch):
    return batch[0]


if __name__ == "__main__":
    # paths & settings
    DS_ROOT = "ds_manipulations/dataset/WOT-20250522(auto)"
    ANN_FILE = "ds_manipulations/annotations-WOT-20250522(auto).json"
    SEQ_LEN = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) dataset
    ds = SequenceDataset(DS_ROOT, ANN_FILE, SEQ_LEN, processor=None)
    ds.processor = AutoImageProcessor.from_pretrained(
        "ustc-community/dfine-small-coco", use_fast=True
    )

    # 2) defect-only id2label + “no_object” at the end
    id2label = {i: lbl for lbl, i in ds.label_map.items()}
    num_defects = len(id2label)
    id2label[num_defects] = "no_object"  # MUST match num_labels = num_defects+1

    # 3) model
    model = TemporalDFine(
        checkpoint="ustc-community/dfine-small-coco",
        num_defect_classes=num_defects,
        id2label=id2label,
    ).to(DEVICE)

    # 4) optimizer: temporal encoder + classifier
    trainable = list(model.temporal_encoder.parameters()) + list(
        model.dfine.class_embed[-1].parameters()
    )
    optimizer = torch.optim.Adam(trainable, lr=1e-4)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=identity_collate,
    )

    # 5) train
    history = {"batch_loss": [], "epoch_loss": []}
    for epoch in range(3):
        model.train()
        running, count = 0.0, 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            pv, sz, tg = (
                batch["pixel_values"].squeeze(0).to(DEVICE),
                batch["sizes"],
                batch["targets"],
            )
            loss, _ = model(pv, sz, targets=tg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            running += loss.item()
            count += 1
            print(f"Batch loss: {loss.item():.4f}")
            history["batch_loss"].append(loss.item())

        avg = running / count
        history["epoch_loss"].append(avg)
        print(f"↳ Epoch {epoch} avg loss {avg:.4f}")

        torch.save(model.state_dict(), f"temporal_dfine_epoch_{epoch}.pth")
        with open("training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("Training complete.")