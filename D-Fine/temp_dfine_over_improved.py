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
                # Skip invalid boxes (zero width/height or out of bounds)
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
                    continue
                boxes.append([(x + w/2)/W, (y + h/2)/H, w/W, h/H])
                labels.append(self.label_map[o["label"]])

            if boxes:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
                
                # Additional validation to ensure no NaN or Inf values
                if torch.isnan(boxes).any() or torch.isinf(boxes).any():
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            targets.append({"boxes": boxes, "class_labels": labels})

        return {"pixel_values": pv, "sizes": sizes, "targets": targets}


# ────────────────────────────────────────────────────────────────────────────────
# 2) Improved TemporalDFine with anomaly detection approach
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
        self.class_head = self.dfine.class_embed[-1]

        # 3) freeze backbone but not the decoder parts
        for name, param in self.dfine.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # 4) temporal encoder with self-attention to learn sequence patterns
        d_model = self.dfine.config.hidden_size
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=4)
        
        # 5) Add a temporal attention mechanism to focus on anomalies
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 6) Add a feature comparison module to detect anomalies
        self.anomaly_detector = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_defect_classes)
        )

        # 7) Initialize weights for new components
        for m in [self.temporal_encoder, self.temporal_attention, self.anomaly_detector]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

        # 8) carry over bbox head & integral logic
        self.bbox_head = self.dfine.bbox_embed[-1]
        self.integral = self.dfine.model.decoder.integral
        self.up = self.dfine.model.decoder.up
        self.reg_scale = self.dfine.config.reg_scale
        self.max_num_bins = self.dfine.config.max_num_bins
        
        # 9) Add a sequence context aggregator
        self.context_aggregator = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.context_projector = nn.Linear(d_model * 2, d_model)

    def forward(self, pixel_values, sizes, targets=None):
        # 1) Extract features with D-Fine backbone
        out = self.dfine.model(pixel_values=pixel_values)
        feats = out.last_hidden_state  # (T, Q, D)
        init_ref = out.init_reference_points  # (T, Q, 2)
        
        batch_size, seq_len, num_queries, d_model = feats.shape[0], feats.shape[0], feats.shape[1], feats.shape[2]
        
        # 2) Reshape for temporal processing
        feats_flat = feats.view(seq_len, num_queries, d_model)  # (T, Q, D)
        
        # 3) Apply temporal encoder to learn sequence patterns
        fused = self.temporal_encoder(feats_flat)  # (T, Q, D)
        
        # 4) Compute temporal attention scores
        attention_scores = self.temporal_attention(fused)  # (T, Q, 1)
        attention_weights = F.softmax(attention_scores, dim=0)  # Normalize across temporal dimension
        
        # 5) Compute sequence context using bidirectional GRU
        context_feats, _ = self.context_aggregator(fused.view(1, seq_len * num_queries, d_model))
        context_feats = self.context_projector(context_feats).view(seq_len, num_queries, d_model)
        
        # 6) Combine context-aware features with attention
        enhanced_feats = fused * attention_weights + context_feats
        
        # 7) Generate class logits and box predictions
        cls_logits = self.class_head(enhanced_feats)  # (T, Q, C+1)
        
        # 8) Apply anomaly detection to boost defect class scores
        anomaly_scores = self.anomaly_detector(enhanced_feats)  # (T, Q, num_defect_classes)
        
        # 9) Combine class logits with anomaly scores
        # Keep no_object class as is, but enhance defect class scores
        defect_enhanced_logits = cls_logits[:, :, :-1] + anomaly_scores
        cls_logits = torch.cat([defect_enhanced_logits, cls_logits[:, :, -1:]], dim=-1)
        
        # 10) Apply safety clamps and handle NaN values
        cls_logits = cls_logits.clamp(-20, 20)
        cls_logits = torch.nan_to_num(
            cls_logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )

        # 11) Generate bounding box predictions
        bbox_dist = self.bbox_head(enhanced_feats)  # (T, Q, 4*(bins+1))
        
        # Handle potential NaN values in bbox_dist
        bbox_dist = torch.nan_to_num(
            bbox_dist,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        # 12) Build projection vector
        Wn = torch.arange(
            self.max_num_bins + 1,
            device=bbox_dist.device,
            dtype=bbox_dist.dtype,
        ) * self.up / self.reg_scale  # (bins+1,)

        # 13) Convert distributions → distances
        distances = self.integral(bbox_dist, Wn)  # (T, Q, 4)
        distances = torch.nan_to_num(distances, nan=0.0, posinf=1.0, neginf=0.0)

        # 14) Turn distances → normalized boxes
        bbox_pred = distance2bbox(init_ref, distances, self.reg_scale).clamp(0, 1)
        bbox_pred = torch.nan_to_num(bbox_pred, nan=0.5, posinf=1.0, neginf=0.0)

        # 15) Inference mode
        if targets is None:
            results = []
            for i, sz in enumerate(sizes):
                res = self.processor.post_process_object_detection(
                    {"logits": cls_logits[i:i + 1], "pred_boxes": bbox_pred[i:i + 1]},
                    target_sizes=torch.tensor([sz], device=cls_logits.device),
                    threshold=0.3,
                )[0]
                results.append(res)
            return results

        # 16) Training loss
        losses = []
        noobj = cls_logits.size(-1) - 1
        
        # 17) Calculate anomaly consistency loss to encourage temporal consistency
        anomaly_consistency_loss = 0.0
        if seq_len > 1:
            # Calculate temporal consistency loss for anomaly scores
            for t in range(1, seq_len):
                anomaly_consistency_loss += F.mse_loss(
                    anomaly_scores[t], anomaly_scores[t-1]
                )
            anomaly_consistency_loss /= (seq_len - 1)
        
        # 18) Process each frame in the sequence
        for i, gt in enumerate(targets):
            pl = cls_logits[i: i + 1]
            pb = bbox_pred[i: i + 1]

            if gt["boxes"].numel() > 0:
                # Check for NaN or Inf values in boxes
                has_invalid_boxes = torch.isnan(gt["boxes"]).any() or torch.isinf(gt["boxes"]).any()
                
                if not has_invalid_boxes:
                    try:
                        # full Hungarian + CE + L1 + GIoU
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
                            initial_reference_points=init_ref[i: i + 1].unsqueeze(0),
                        )
                    except Exception as e:
                        # More general exception handling - catch any error in the matcher
                        print(f"Warning: Matching failed with error: {e}. Using fallback loss for frame {i}.")
                        logp = pl.squeeze(0)
                        fallback = torch.full((logp.size(0),), noobj,
                                            dtype=torch.long, device=logp.device)
                        lc = F.cross_entropy(logp, fallback)
                else:
                    # Invalid boxes detected, use fallback
                    print(f"Warning: Invalid box values detected in frame {i}. Using fallback loss.")
                    logp = pl.squeeze(0)
                    fallback = torch.full((logp.size(0),), noobj,
                                        dtype=torch.long, device=logp.device)
                    lc = F.cross_entropy(logp, fallback)
            else:
                # No boxes in ground truth, use fallback
                logp = pl.squeeze(0)
                fallback = torch.full((logp.size(0),), noobj,
                                    dtype=torch.long, device=logp.device)
                lc = F.cross_entropy(logp, fallback)

            losses.append(lc)
        
        # 19) Combine all losses
        detection_loss = torch.stack(losses).mean()
        total_loss = detection_loss + 0.1 * anomaly_consistency_loss
        
        return total_loss, {"detection_loss": detection_loss.item(), 
                           "anomaly_consistency_loss": anomaly_consistency_loss.item()}


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

    # 2) defect-only id2label + "no_object" at the end
    id2label = {i: lbl for lbl, i in ds.label_map.items()}
    num_defects = len(id2label)
    id2label[num_defects] = "no_object"  # MUST match num_labels = num_defects+1
    print(f"Training with {num_defects} defect classes: {list(id2label.values())[:-1]}")

    # 3) model
    model = TemporalDFine(
        checkpoint="ustc-community/dfine-small-coco",
        num_defect_classes=num_defects,
        id2label=id2label,
    ).to(DEVICE)

    # 4) optimizer: all trainable parameters with different learning rates
    backbone_params = []
    temporal_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'temporal' in name or 'anomaly' in name or 'context' in name:
                temporal_params.append(param)
            elif 'class' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': temporal_params, 'lr': 5e-4},
        {'params': classifier_params, 'lr': 1e-4}
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=identity_collate,
    )

    # 5) train
    history = {"batch_loss": [], "epoch_loss": [], "detection_loss": [], "anomaly_consistency_loss": []}
    for epoch in range(10):  # Increased epochs for better learning
        model.train()
        running, count = 0.0, 0
        epoch_detection_loss = 0.0
        epoch_anomaly_loss = 0.0
        
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            try:
                pv, sz, tg = (
                    batch["pixel_values"].squeeze(0).to(DEVICE),
                    batch["sizes"],
                    batch["targets"],
                )
                
                # Validate targets before processing
                valid_batch = True
                for t in tg:
                    if t["boxes"].numel() > 0:
                        if torch.isnan(t["boxes"]).any() or torch.isinf(t["boxes"]).any():
                            print(f"Warning: Skipping batch with invalid box values")
                            valid_batch = False
                            break
                
                if not valid_batch:
                    continue
                    
                loss, loss_dict = model(pv, sz, targets=tg)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running += loss.item()
                epoch_detection_loss += loss_dict["detection_loss"]
                epoch_anomaly_loss += loss_dict["anomaly_consistency_loss"]
                count += 1
                print(f"Batch loss: {loss.item():.4f} (Detection: {loss_dict['detection_loss']:.4f}, Anomaly: {loss_dict['anomaly_consistency_loss']:.4f})")
                history["batch_loss"].append(loss.item())
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

        scheduler.step()
        
        avg = running / count
        avg_detection = epoch_detection_loss / count
        avg_anomaly = epoch_anomaly_loss / count
        
        history["epoch_loss"].append(avg)
        history["detection_loss"].append(avg_detection)
        history["anomaly_consistency_loss"].append(avg_anomaly)
        
        print(f"↳ Epoch {epoch} avg loss {avg:.4f} (Detection: {avg_detection:.4f}, Anomaly: {avg_anomaly:.4f})")

        # Save model checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg,
            "id2label": id2label,
            "num_defect_classes": num_defects
        }
        torch.save(checkpoint, f"temporal_dfine_improved_epoch_{epoch}.pth")
        
        # Save training history
        with open("training_history_improved.json", "w") as f:
            json.dump(history, f, indent=2)

    print("Training complete.")
