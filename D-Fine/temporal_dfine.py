#
import os
import json
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
from transformers.models.d_fine.modeling_d_fine import (
    weighting_function,
    distance2bbox,
)
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# 1) Sequence‐of‐50 Dataset
# ────────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(
        self,
        ds_root="ds_manipulations/dataset/WOT-20250522(auto)",
        ann_file="ds_manipulations/annotations-WOT-20250522(auto).json",
        seq_len=50,
        processor=None,
        label_map=None,
    ):
        # load annotations JSON
        with open(ann_file, "r") as f:
            self.anns = json.load(f)

        # if no label_map provided, scan through all anns and build one
        if label_map is None:
            all_labels = {
                obj["label"]
                for folder in self.anns.values()
                for annots in folder.values()
                for obj in annots
            }
            # assign each unique string label an integer ID
            self.label_map = {lab: i for i, lab in enumerate(sorted(all_labels))}
        else:
            self.label_map = label_map

        self.seq_len   = seq_len
        self.processor = processor

        # build a list of sequences exactly as before...
        self.sequences = []
        for folder in sorted(os.listdir(ds_root)):
            fp = os.path.join(ds_root, folder)
            if not os.path.isdir(fp): continue
            imgs = [os.path.join(fp, fn) for fn in os.listdir(fp) if fn.endswith(".png")]
            imgs.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
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

        filtered = []
        for seq in self.sequences:
            folder = os.path.basename(os.path.dirname(seq[0]))
            if any(self.anns[folder][os.path.basename(p)] for p in seq):
                filtered.append(seq)
        self.sequences = filtered

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        paths = self.sequences[idx]
        imgs  = [Image.open(p).convert("RGB") for p in paths]

        proc = self.processor(images=imgs, return_tensors="pt")
        pv   = proc["pixel_values"]  # (seq_len,3,H,W)

        sizes, targets = [], []
        for p, im in zip(paths, imgs):
            sizes.append(im.size[::-1])
            folder = os.path.basename(os.path.dirname(p))
            fn     = os.path.basename(p)
            annots = self.anns.get(folder, {}).get(fn, [])

            boxes, labels = [], []
            W, H = im.size
            for obj in annots:
                x1, x2, y1, y2 = obj["bbox"]
                x, y = min(x1,x2), min(y1,y2)
                w, h = abs(x2-x1), abs(y2-y1)
                # normalize to [0,1]
                cx = (x + w/2)/W
                cy = (y + h/2)/H
                boxes.append([cx, cy, w/W, h/H])
                labels.append(self.label_map[obj["label"]])

            if boxes:
                boxes  = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
            else:
                boxes  = torch.zeros((0,4), dtype=torch.float32)
                labels = torch.zeros((0,),   dtype=torch.int64)

            targets.append({"boxes": boxes, "class_labels": labels})

        return {"pixel_values": pv, "sizes": sizes, "targets": targets}


# ────────────────────────────────────────────────────────────────────────────────
# 2) TemporalDFine + training
# ────────────────────────────────────────────────────────────────────────────────

class TemporalDFine(nn.Module):
    def __init__(self, checkpoint="ustc-community/dfine-small-coco"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
        self.dfine     = DFineForObjectDetection.from_pretrained(checkpoint)

        # ─── freeze DFine weights & disable its dropout ───────────────────────
        for p in self.dfine.parameters():
            p.requires_grad = False
        self.dfine.eval()

        # ─── build & tiny-init your temporal transformer ──────────────────────
        d_model = self.dfine.config.hidden_size
        layer   = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=4)

        for m in self.temporal_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # ─── heads & integral logic as before ────────────────────────────────
        self.class_head   = self.dfine.class_embed[-1]
        self.bbox_head    = self.dfine.bbox_embed[-1]
        self.integral     = self.dfine.model.decoder.integral
        self.up           = self.dfine.model.decoder.up
        self.reg_scale    = self.dfine.config.reg_scale
        self.max_num_bins = self.dfine.config.max_num_bins

    def forward(self, pixel_values, sizes, targets=None):
        # 1) extract per-frame features & initial references
        out = self.dfine.model(pixel_values=pixel_values)
        feats = out.last_hidden_state  # (T, Q, D)
        init_ref = out.init_reference_points  # (T, Q, 4)

        # 2) fuse across T frames
        fused = self.temporal_encoder(feats)  # (T, Q, D)

        # 3) run the classification & bbox-distribution heads
        cls_logits = self.class_head(fused)  # (T, Q, C+1)
        cls_logits = cls_logits.clamp(-20, 20)  # <— avoid extreme logits
        bbox_dist = self.bbox_head(fused)  # (T, Q, 4*(bins+1))

        # 4) turn distributions → normalized [0,1] boxes
        proj = weighting_function(
            self.max_num_bins,
            self.up,
            self.dfine.model.decoder.reg_scale,
        )
        distances = self.integral(bbox_dist, proj)  # (T, Q, 4)
        bbox_pred = distance2bbox(init_ref, distances, self.reg_scale).clamp(0, 1)

        # 5) if no targets passed, just return D-Fine’s usual post-processed preds
        if targets is None:
            results = []
            for i, sz in enumerate(sizes):
                out_i = self.processor.post_process_object_detection(
                    {"logits": cls_logits[i:i + 1], "pred_boxes": bbox_pred[i:i + 1]},
                    target_sizes=torch.tensor([sz]),
                    threshold=0.3,
                )[0]
                results.append(out_i)
            return results

        # 6) otherwise build one scalar loss averaged over all T frames
        losses = []
        noobj = cls_logits.size(-1) - 1
        for i, gt in enumerate(targets):
            # per-frame predictions
            pred_logits = cls_logits[i:i + 1]  # (1, Q, C+1)
            pred_boxes = bbox_pred[i:i + 1]  # (1, Q, 4)

            if gt["boxes"].numel() > 0:
                try:
                    # try the full D-Fine Hungarian + CE + L1 + GIoU loss
                    frame_loss, _, _ = self.dfine.loss_function(
                        logits=pred_logits,
                        labels=[gt],
                        device=pixel_values.device,
                        pred_boxes=pred_boxes,
                        config=self.dfine.config,
                        outputs_class=pred_logits.unsqueeze(0),
                        outputs_coord=pred_boxes.unsqueeze(0),
                        enc_topk_logits=None,
                        enc_topk_bboxes=None,
                        denoising_meta_values=None,
                        predicted_corners=None,
                        initial_reference_points=init_ref[i:i + 1].unsqueeze(0),
                    )
                except ValueError:
                    # fallback: treat as empty frame
                    logp = pred_logits.squeeze(0)  # (Q, C+1)
                    tgt = torch.full((logp.size(0),), noobj,
                                     dtype=torch.long, device=logp.device)
                    frame_loss = F.cross_entropy(logp, tgt)
            else:
                # empty frame → push all queries to “no-object”
                logp = pred_logits.squeeze(0)
                tgt = torch.full((logp.size(0),), noobj,
                                 dtype=torch.long, device=logp.device)
                frame_loss = F.cross_entropy(logp, tgt)

            losses.append(frame_loss)

        # average and return (loss, empty_dict) so your training loop stays happy
        total_loss = torch.stack(losses).mean()
        return total_loss, {}


# ────────────────────────────────────────────────────────────────────────────────
# 3) put it all together
# ────────────────────────────────────────────────────────────────────────────────

def identity_collate(batch):
    # Dataset returns a single dict per item,
    # so just unwrap the one‐element list
    return batch[0]

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalDFine().to(DEVICE)
    ds = SequenceDataset(
        processor=model.processor,
        ds_root="ds_manipulations/dataset/WOT-20250522(auto)",
        ann_file="ds_manipulations/annotations-WOT-20250522(auto).json",
        seq_len=50,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=identity_collate
    )
    print("Total num batches is ", len(loader))
    optimizer = torch.optim.Adam(model.temporal_encoder.parameters(), lr=1e-5)

    history = {"epoch_loss": []}

    for epoch in range(10):
        model.train()
        # model.dfine.eval()
        running_loss = 0.0
        n_batches   = 0

        for batch in tqdm(loader):
            pv = batch["pixel_values"].squeeze(0).to(DEVICE)
            sizes = batch["sizes"]
            targets = batch["targets"]

            loss, _ = model(pv, sizes, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.temporal_encoder.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            running_loss += loss.item()
            n_batches   += 1
            print(f"Batch loss: {loss}")

        avg_loss = running_loss / n_batches
        history["epoch_loss"].append(avg_loss)
        print(f"________ Epoch {epoch} — avg loss: {avg_loss:.4f} ______")

        # checkpoint after each epoch
        ckpt_path = f"temporal_dfine_epoch_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)

        # save history so far
        with open("training_history.json", "w") as hf:
            json.dump(history, hf, indent=2)

    # final save
    torch.save(model.state_dict(), "temporal_dfine_final.pth")
    with open("training_history.json", "w") as hf:
        json.dump(history, hf, indent=2)

    print("Training complete. Final model saved to temporal_dfine_final.pth")
    print("Loss history saved to training_history.json")