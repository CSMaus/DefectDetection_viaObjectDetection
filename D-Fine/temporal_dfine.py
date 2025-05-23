#
import os
import json
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
from transformers.models.d_fine.modeling_d_fine import (
    weighting_function,
    distance2bbox,
)

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

        # freeze entire DFine except our transformer
        for p in self.dfine.parameters():
            p.requires_grad = False

        d_model = self.dfine.config.hidden_size
        layer   = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=4)

        # heads & integral
        self.class_head   = self.dfine.class_embed[-1]
        self.bbox_head    = self.dfine.bbox_embed[-1]
        self.integral     = self.dfine.model.decoder.integral
        self.up           = self.dfine.model.decoder.up
        self.reg_scale    = self.dfine.config.reg_scale
        self.max_num_bins = self.dfine.config.max_num_bins

    def forward(self, pixel_values, sizes, targets=None):
        # pixel_values: (50,3,H,W), sizes: list[50 of (H,W)]
        out   = self.dfine.model(pixel_values=pixel_values)
        feats = out.last_hidden_state               # (50, Q, D)
        init_ref = out.init_reference_points        # (50, Q, 4)

        # fuse across the 50 frames:
        fused = self.temporal_encoder(feats)        # (50, Q, D)

        cls_logits = self.class_head(fused)         # (50, Q, #classes)
        bbox_dist  = self.bbox_head(fused)          # (50, Q, 4*(bins+1))

        Wn = weighting_function(self.max_num_bins,
                                self.up,
                                self.dfine.model.decoder.reg_scale)
        distances = self.integral(bbox_dist, Wn)    # (50, Q, 4)
        bbox_pred  = distance2bbox(init_ref, distances, self.reg_scale)

        # if no targets, just return raw preds:
        if targets is None:
            results = []
            for i, sz in enumerate(sizes):
                r = self.processor.post_process_object_detection(
                        {"logits": cls_logits[i:i+1], "pred_boxes": bbox_pred[i:i+1]},
                        target_sizes=torch.tensor([sz]),
                        threshold=0.3
                    )[0]
                results.append(r)
            return results

        # otherwise compute the DFine loss (Hungarian + CE + L1 + GIoU):
        # wrap logits/boxes into the format DFineForObjectDetection expects
        outputs_class = cls_logits.unsqueeze(0)     # pretend 1 layer of intermediate
        outputs_coord = bbox_pred.unsqueeze(0)
        init_refs     = init_ref.unsqueeze(0)

        loss, loss_dict, _ = self.dfine.loss_function(
            logits                      = cls_logits,      # (50,Q,C)
            labels                      = targets,         # List[50] of dicts
            device                      = pixel_values.device,
            pred_boxes                  = bbox_pred,       # (50,Q,4)
            config                      = self.dfine.config,
            outputs_class               = outputs_class,   # (1,50,Q,C)
            outputs_coord               = outputs_coord,   # (1,50,Q,4)
            enc_topk_logits             = None,
            enc_topk_bboxes             = None,
            denoising_meta_values       = None,
            predicted_corners           = None,
            initial_reference_points    = init_refs,       # (1,50,Q,4)
        )
        return loss, loss_dict


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
        batch_size=1,  # one 50‐frame sequence per batch
        shuffle=True,
        num_workers=4,  # now safe because collate_fn is a top‐level function
        pin_memory=True,
        collate_fn=identity_collate
    )

    optimizer = torch.optim.Adam(model.temporal_encoder.parameters(), lr=1e-4)

    for epoch in range(10):
        batch_idx = 0
        for batch in loader:
            # batch is a dict with keys: "pixel_values", "sizes", "targets"
            pv      = batch["pixel_values"].squeeze(0).to(DEVICE)  # (50,3,H,W)
            sizes   = batch["sizes"]                               # list of 50 (H,W)
            targets = batch["targets"]                    # list of 50 dicts

            loss, loss_dict = model(pv, sizes, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}: iter {batch_idx} — loss: {loss.item():.4f}")
            batch_idx += 1

    '''
    # instantiate model + dataset
    tmp = TemporalDFine().to(DEVICE)
    ds  = SequenceDataset(
        processor=tmp.processor,
        label_map=tmp.dfine.config.label2id
    )
    loader = DataLoader(
        ds,
        batch_size=1,        # one 50‐frame sequence per batch
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: x[0]  # we return a single dict per batch
    )

    opt = torch.optim.Adam(tmp.temporal_encoder.parameters(), lr=1e-4)

    for epoch in range(10):
        for batch in loader:
            pv      = batch["pixel_values"].squeeze(0).to(DEVICE)
            sizes   = batch["sizes"]
            targets = batch["targets"][0]  # list of 50 dicts

            loss, loss_dict = tmp(pv, sizes, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"Epoch {epoch}  loss = {loss.item():.4f}")
    '''