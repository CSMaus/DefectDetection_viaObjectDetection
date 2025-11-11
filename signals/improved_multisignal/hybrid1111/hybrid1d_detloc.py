# hybrid1d_detloc.py
# Full, drop‑in model + losses + training loop for 1D signal SEQUENCES with
# detection (objectness) + localization (center, width) using:
#  - High‑res 1D Conv backbone + FPN (multi‑scale along signal_length)
#  - Sequence‑context Transformer across the sequence dimension (num_signals)
#  - Separate heads for cls/reg (no shared last layer)
#  - Anchor‑free, center‑based targets (Gaussian heatmap like CenterNet, but 1D)
#  - Stable losses: BCE‑with‑logits (+Focal optional) for obj, L1 + IoU1D for boxes
#
# Expected batch from dataloader:
#   signals: FloatTensor [B, N, S]
#   labels:  FloatTensor [B, N] in {0,1}
#   meta (optional dict or tensor(s)) may include:
#       centers: FloatTensor [B, N] in [0,1]  (normalized center position along S)
#       widths:  FloatTensor [B, N] in [0,1]  (normalized width/extent)
# If centers/widths are missing, training runs detection‑only.
#
# Inference per signal returns best center/width from highest obj position at the finest scale.

from __future__ import annotations
import math
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


# ------------------------------
# Utils: 1D IoU, Focal BCE
# ------------------------------

def interval_iou1d(c1, w1, c2, w2):
    """IoU for 1D segments parameterized by center c and width w (all in [0,1]).
    Returns IoU in [0,1]. Shapes: broadcastable.
    """
    x1_l = c1 - 0.5 * w1
    x1_r = c1 + 0.5 * w1
    x2_l = c2 - 0.5 * w2
    x2_r = c2 + 0.5 * w2
    inter_l = torch.maximum(x1_l, x2_l)
    inter_r = torch.minimum(x1_r, x2_r)
    inter = torch.clamp(inter_r - inter_l, min=0.0)
    union = (x1_r - x1_l) + (x2_r - x2_l) - inter + 1e-8
    return inter / union


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: any shape, raw scores. targets in {0,1}
        prob = torch.sigmoid(logits)
        p_t = targets * prob + (1 - targets) * (1 - prob)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = alpha_t * (1 - p_t) ** self.gamma * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ------------------------------
# Backbone: 1D Conv + FPN
# ------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, act=True):
        super().__init__()
        if p is None:
            p = (k // 2) * d
        self.conv = nn.Conv1d(c_in, c_out, k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock1D(nn.Module):
    def __init__(self, c, n=2):
        super().__init__()
        self.blocks = nn.Sequential(*[nn.Sequential(ConvBNAct(c, c, 3), ConvBNAct(c, c, 3)) for _ in range(n)])
    def forward(self, x):
        return x + self.blocks(x)


class BackboneFPN1D(nn.Module):
    """
    Produce multi‑scale feature maps along the signal axis.
    Input: [B*N, 1, S]
    Outputs: dict {"P3": [B*N, C, S/8], "P4": [B*N, C, S/16], "P5": [B*N, C, S/32]}
    """
    def __init__(self, c=128):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(1, 32, 3, 1),
            ConvBNAct(32, 64, 3, 2),  # /2
            CSPBlock1D(64, n=1),
            ConvBNAct(64, 128, 3, 2),  # /4
            CSPBlock1D(128, n=1),
        )
        # Down path to /8, /16, /32
        self.down8 = nn.Sequential(ConvBNAct(128, 128, 3, 2), CSPBlock1D(128, n=2))   # /8
        self.down16 = nn.Sequential(ConvBNAct(128, 192, 3, 2), CSPBlock1D(192, n=2))  # /16
        self.down32 = nn.Sequential(ConvBNAct(192, 256, 3, 2), CSPBlock1D(256, n=2))  # /32
        # Lateral for FPN
        self.lat8 = ConvBNAct(128, c, 1, act=False)
        self.lat16 = ConvBNAct(192, c, 1, act=False)
        self.lat32 = ConvBNAct(256, c, 1, act=False)
        self.smooth8 = ConvBNAct(c, c, 3)
        self.smooth16 = ConvBNAct(c, c, 3)
        self.smooth32 = ConvBNAct(c, c, 3)

    def forward(self, x):
        # x: [B*N,1,S]
        x = self.stem(x)
        c8 = self.down8(x)
        c16 = self.down16(c8)
        c32 = self.down32(c16)
        p32 = self.smooth32(self.lat32(c32))
        up16 = F.interpolate(p32, size=c16.shape[-1], mode='linear', align_corners=False)
        p16 = self.smooth16(self.lat16(c16) + up16)
        up8 = F.interpolate(p16, size=c8.shape[-1], mode='linear', align_corners=False)
        p8 = self.smooth8(self.lat8(c8) + up8)
        return {"P3": p8, "P4": p16, "P5": p32}


# ------------------------------
# Sequence context transformer (across N signals)
# ------------------------------

class SeqContextTransformer(nn.Module):
    def __init__(self, in_dim=128, ctx_dim=128, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=ctx_dim, nhead=num_heads, dim_feedforward=4*ctx_dim,
                                               dropout=dropout, batch_first=True, activation='gelu')
        self.proj_in = nn.Linear(in_dim, ctx_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(ctx_dim, in_dim)

    def forward(self, per_signal_global: torch.Tensor):
        # per_signal_global: [B,N,in_dim]; returns [B,N,in_dim]
        z = self.proj_in(per_signal_global)
        z = self.encoder(z)
        return self.proj_out(z)


class FiLM1D(nn.Module):
    # Feature‑wise linear modulation for conditioning P3/P4/P5 by sequence context
    def __init__(self, ctx_dim, feat_dim):
        super().__init__()
        self.gamma = nn.Linear(ctx_dim, feat_dim)
        self.beta = nn.Linear(ctx_dim, feat_dim)
    def forward(self, feat: torch.Tensor, ctx: torch.Tensor):
        # feat: [B*N, C, L], ctx: [B*N, C]
        g = self.gamma(ctx).unsqueeze(-1)
        b = self.beta(ctx).unsqueeze(-1)
        return feat * (1 + torch.tanh(g)) + b


# ------------------------------
# Heads: separate cls / reg towers (anchor‑free)
# ------------------------------

class Head1D(nn.Module):
    def __init__(self, c_in, n_cls=1, n_reg=2):
        super().__init__()
        tower_c = [ConvBNAct(c_in, c_in, 3) for _ in range(3)]
        tower_r = [ConvBNAct(c_in, c_in, 3) for _ in range(3)]
        self.tower_cls = nn.Sequential(*tower_c)
        self.tower_reg = nn.Sequential(*tower_r)
        self.cls_logits = nn.Conv1d(c_in, n_cls, 1)
        self.reg_params = nn.Conv1d(c_in, n_reg, 1)  # center_offset, log_width

    def forward(self, x):
        c = self.cls_logits(self.tower_cls(x))           # [B*N,1,L]
        r = self.reg_params(self.tower_reg(x))           # [B*N,2,L]
        return c, r


# ------------------------------
# Full Model
# ------------------------------

class Hybrid1D_DetLoc(nn.Module):
    """
    Input:  signals [B,N,S]
    Output (logits): dict with per‑scale tensors (no sigmoid in model):
        obj_logits_{P3,P4,P5}: [B,N,L]
        reg_{P3,P4,P5}:       [B,N,2,L]  (center_offset, log_width) in cell coordinates
    Inference helper provided below.
    """
    def __init__(self, signal_length=320, fpn_dim=128, ctx_dim=128, num_heads=8, ctx_layers=2):
        super().__init__()
        self.signal_length = signal_length
        self.backbone = BackboneFPN1D(c=fpn_dim)
        self.head = Head1D(fpn_dim)
        # global pooling per signal for context
        self.per_signal_pool = nn.AdaptiveAvgPool1d(1)
        self.seq_ctx = SeqContextTransformer(in_dim=fpn_dim, ctx_dim=ctx_dim, num_heads=num_heads, num_layers=ctx_layers)
        self.film_p3 = FiLM1D(ctx_dim=fpn_dim, feat_dim=fpn_dim)
        self.film_p4 = FiLM1D(ctx_dim=fpn_dim, feat_dim=fpn_dim)
        self.film_p5 = FiLM1D(ctx_dim=fpn_dim, feat_dim=fpn_dim)

    def _merge_bn(self, x_dict, B, N):
        # dict of {Pi: [B*N,C,L]} -> also compute per‑signal pooled features and context
        P3, P4, P5 = x_dict["P3"], x_dict["P4"], x_dict["P5"]
        # compute per‑signal pooled features from P3 (highest resolution)
        C = P3.shape[1]
        pooled = self.per_signal_pool(P3).squeeze(-1)     # [B*N,C]
        pooled = pooled.view(B, N, C)
        ctx = self.seq_ctx(pooled)                        # [B,N,C]
        ctx_bn = ctx.view(B*N, C)
        # FiLM conditioning
        P3 = self.film_p3(P3, ctx_bn)
        P4 = self.film_p4(P4, ctx_bn)
        P5 = self.film_p5(P5, ctx_bn)
        return {"P3": P3, "P4": P4, "P5": P5}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, S = x.shape
        x = x.view(B * N, 1, S)
        fpn = self.backbone(x)                      # {P3,P4,P5}: [B*N,C,L]
        fpn = self._merge_bn(fpn, B, N)             # sequence‑conditioned features
        out = {}
        for k in ("P3", "P4", "P5"):
            obj, reg = self.head(fpn[k])            # [B*N,1,L], [B*N,2,L]
            # reshape back to [B,N,L] and [B,N,2,L]
            out[f"obj_logits_{k}"] = obj.view(B, N, -1)
            out[f"reg_{k}"] = reg.view(B, N, 2, -1)
        return out

    @torch.no_grad()
    def infer_single(self, x: torch.Tensor) -> Dict[str, Any]:
        self.eval()
        out = self.forward(x)
        obj = torch.sigmoid(out['obj_logits_P3'])  # [1,N,L]
        reg = out['reg_P3']  # [1,N,2,L]
        assert reg.dim() == 4 and reg.shape[2] == 2, f"reg head must be [1,N,2,L], got {tuple(reg.shape)}"

        probs, idx = obj.max(dim=-1)  # [1,N]
        idx_exp = idx.unsqueeze(-1)  # [1,N,1]
        L = obj.shape[-1]

        reg_c = reg[:, :, 0, :]  # [1,N,L]
        reg_w = reg[:, :, 1, :]  # [1,N,L]

        center_off = torch.tanh(reg_c.gather(-1, idx_exp).squeeze(-1))  # [1,N]
        logw = reg_w.gather(-1, idx_exp).squeeze(-1)  # [1,N]

        cell_ctr = (idx.float() + center_off + 0.5) / L
        width = (logw.exp() / L).clamp(1e-4, 1.0)
        return {"prob": probs.squeeze(0), "center": cell_ctr.squeeze(0), "width": width.squeeze(0)}


# ------------------------------
# Targets & Losses for anchor‑free 1D
# ------------------------------

@dataclass
class DetLocLossCfg:
    lambda_obj: float = 1.0
    lambda_l1: float = 1.0
    lambda_iou: float = 1.0
    focal: bool = True
    gaussian_sigma: float = 2.0  # in cells


class DetLocCriterion(nn.Module):
    def __init__(self, cfg: DetLocLossCfg):
        super().__init__()
        self.cfg = cfg
        self.obj_loss = FocalBCEWithLogitsLoss() if cfg.focal else nn.BCEWithLogitsLoss()
        self.l1 = nn.SmoothL1Loss(reduction='none')

    def _make_heatmap(self, L: int, centers01: torch.Tensor, mask: torch.Tensor):
        device = centers01.device
        idx = (centers01 * L).clamp(0, L - 1e-4)
        grid = torch.arange(L, device=device).view(1, 1, L).float()
        sigma = self.cfg.gaussian_sigma
        heat = torch.exp(-0.5 * ((grid - idx.unsqueeze(-1)) / sigma) ** 2)
        return heat * mask.unsqueeze(-1)

    def forward(self, out: Dict[str, torch.Tensor], labels: torch.Tensor,
                centers01: Optional[torch.Tensor], widths01: Optional[torch.Tensor]):
        device = labels.device
        # keep them as tensors so .item() always works upstream
        total_obj = torch.tensor(0.0, device=device)
        total_l1  = torch.tensor(0.0, device=device)
        total_iou = torch.tensor(0.0, device=device)

        B, N = labels.shape
        pos_mask = (labels > 0.5).float()
        any_pos = (pos_mask.sum() > 0)

        for k in ("P3", "P4", "P5"):
            obj_logits = out[f"obj_logits_{k}"]  # [B,N,L]
            L = obj_logits.shape[-1]

            # objectness target
            if any_pos and (centers01 is not None):
                heat = self._make_heatmap(L, centers01, pos_mask)  # [B,N,L]
            else:
                heat = labels.unsqueeze(-1).expand_as(obj_logits)  # [B,N,L]
            total_obj = total_obj + self.obj_loss(obj_logits, heat)

            # localization (only if positives + targets provided)
            if any_pos and (centers01 is not None) and (widths01 is not None):
                reg = out[f"reg_{k}"]  # [B,N,2,L]
                assert reg.dim() == 4 and reg.shape[2] == 2, f"reg head must be [B,N,2,L], got {tuple(reg.shape)}"

                tgt_idx = (centers01 * L).clamp(0, L - 1e-4).long()  # [B,N]
                idx_exp = tgt_idx.unsqueeze(-1)                      # [B,N,1]

                # slice channels first to avoid gather collapsing the 2-channel dim
                reg_c = reg[:, :, 0, :]  # [B,N,L] center_offset
                reg_w = reg[:, :, 1, :]  # [B,N,L] log_width

                off  = torch.tanh(reg_c.gather(-1, idx_exp).squeeze(-1))  # [B,N]
                logw = reg_w.gather(-1, idx_exp).squeeze(-1)              # [B,N]

                pred_c = ((tgt_idx.float() + off + 0.5) / L).clamp(0, 1)
                pred_w = (logw.exp() / L).clamp(1e-4, 1.0)

                l1_c = self.l1(pred_c, centers01)
                l1_w = self.l1(pred_w, widths01)
                l1   = ((l1_c + l1_w) * pos_mask).sum() / (pos_mask.sum() + 1e-8)
                total_l1 = total_l1 + l1

                iou = interval_iou1d(pred_c, pred_w, centers01, widths01)
                iou_loss = ((1.0 - iou) * pos_mask).sum() / (pos_mask.sum() + 1e-8)
                total_iou = total_iou + iou_loss

        loss = (self.cfg.lambda_obj * total_obj +
                self.cfg.lambda_l1  * total_l1  +
                self.cfg.lambda_iou * total_iou)
        return {"loss": loss, "loss_obj": total_obj, "loss_l1": total_l1, "loss_iou": total_iou}


# with error. maybe replace .item() from loses, bcs I think it usually retuns float
class DetLocCriterion_old(nn.Module):
    def __init__(self, cfg: DetLocLossCfg):
        super().__init__()
        self.cfg = cfg
        self.obj_loss = FocalBCEWithLogitsLoss() if cfg.focal else nn.BCEWithLogitsLoss()
        self.l1 = nn.SmoothL1Loss(reduction='none')

    def _make_heatmap(self, L: int, centers01: torch.Tensor, mask: torch.Tensor):
        # centers01: [B,N] normalized in [0,1]; mask: [B,N] {0,1} for positives
        # returns heat: [B,N,L]
        device = centers01.device
        idx = (centers01 * L).clamp(0, L - 1e-4)
        grid = torch.arange(L, device=device).view(1, 1, L).float()
        # gaussian around center index
        sigma = self.cfg.gaussian_sigma
        heat = torch.exp(-0.5 * ((grid - idx.unsqueeze(-1)) / sigma) ** 2)
        heat = heat * mask.unsqueeze(-1)
        return heat

    def forward(self, out: Dict[str, torch.Tensor], labels: torch.Tensor,
                centers01: Optional[torch.Tensor], widths01: Optional[torch.Tensor]):
        # labels: [B,N] 0/1
        total_obj = 0.0
        total_l1 = 0.0
        total_iou = 0.0
        B, N = labels.shape
        pos_mask = (labels > 0.5).float()
        any_pos = pos_mask.sum() > 0
        for k in ("P3", "P4", "P5"):
            obj_logits = out[f"obj_logits_{k}"]  # [B,N,L]
            L = obj_logits.shape[-1]
            # Objectness targets
            if any_pos and (centers01 is not None):
                heat = self._make_heatmap(L, centers01, pos_mask)  # [B,N,L]
            else:
                heat = labels.unsqueeze(-1).expand_as(obj_logits)   # coarse target if no centers
            total_obj = total_obj + self.obj_loss(obj_logits, heat)

            # Localization (if provided)
            # Localization (if provided)
            if any_pos and (centers01 is not None) and (widths01 is not None):
                reg = out[f"reg_{k}"]  # [B,N,2,L] expected
                assert reg.dim() == 4 and reg.shape[2] == 2, f"reg head must be [B,N,2,L], got {tuple(reg.shape)}"
                L = reg.shape[-1]

                tgt_idx = (centers01 * L).clamp(0, L - 1e-4).long()  # [B,N]
                idx_exp = tgt_idx.unsqueeze(-1)  # [B,N,1]

                # Gather per-channel to avoid shape collapse when using gather on last dim
                reg_c = reg[:, :, 0, :]  # [B,N,L]  (center_offset)
                reg_w = reg[:, :, 1, :]  # [B,N,L]  (log_width)

                off = torch.tanh(reg_c.gather(-1, idx_exp).squeeze(-1))  # [B,N]
                logw = reg_w.gather(-1, idx_exp).squeeze(-1)  # [B,N]

                pred_c = ((tgt_idx.float() + off + 0.5) / L).clamp(0, 1)
                pred_w = (logw.exp() / L).clamp(1e-4, 1.0)

                l1_c = self.l1(pred_c, centers01)
                l1_w = self.l1(pred_w, widths01)
                l1 = (l1_c + l1_w) * pos_mask
                l1 = l1.sum() / (pos_mask.sum() + 1e-8)
                total_l1 = total_l1 + l1

                iou = interval_iou1d(pred_c, pred_w, centers01, widths01)
                iou_loss = (1.0 - iou) * pos_mask
                iou_loss = iou_loss.sum() / (pos_mask.sum() + 1e-8)
                total_iou = total_iou + iou_loss

        loss = (self.cfg.lambda_obj * total_obj +
                self.cfg.lambda_l1 * total_l1 +
                self.cfg.lambda_iou * total_iou)
        return {"loss": loss, "loss_obj": total_obj, "loss_l1": total_l1, "loss_iou": total_iou}




# ------------------------------
# Training loop (compatible with detection‑only or det+loc)
# ------------------------------

@dataclass
class TrainCfg:
    lr: float = 8e-4
    weight_decay: float = 1.5e-2
    epochs: int = 15
    clip_grad: float = 1.0


def train_detloc(model: nn.Module,
                 train_loader,
                 val_loader,
                 device: torch.device,
                 save_dir: str,
                 loss_cfg: DetLocLossCfg = DetLocLossCfg(),
                 opt_cfg: TrainCfg = TrainCfg()):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    criterion = DetLocCriterion(loss_cfg).to(device)

    best_val = float('inf')

    for epoch in range(1, opt_cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            # Accept multiple batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    signals, labels, meta = batch
                else:
                    signals, labels = batch
                    meta = None
            elif isinstance(batch, dict):
                signals = batch['signals']
                labels = batch['labels']
                meta = batch
            else:
                raise RuntimeError("Unsupported batch type")

            signals = signals.to(device).float()   # [B,N,S]
            labels = labels.to(device).float()     # [B,N]
            centers = meta.get('centers', None).to(device).float() if (meta is not None and 'centers' in meta) else None
            widths = meta.get('widths', None).to(device).float() if (meta is not None and 'widths' in meta) else None

            out = model(signals)
            losses = criterion(out, labels, centers, widths)

            optimizer.zero_grad()
            losses['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.clip_grad)
            optimizer.step()
            tr_loss += float(losses['loss'].item())

        tr_loss /= max(1, len(train_loader))

        # ---------- validation
        model.eval()
        with torch.no_grad():
            va_loss = 0.0
            for batch in tqdm(val_loader, desc="Val"):
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        signals, labels, meta = batch
                    else:
                        signals, labels = batch
                        meta = None
                else:
                    signals = batch['signals']; labels = batch['labels']; meta = batch
                signals = signals.to(device).float()
                labels = labels.to(device).float()
                centers = meta.get('centers', None).to(device).float() if (meta is not None and 'centers' in meta) else None
                widths = meta.get('widths', None).to(device).float() if (meta is not None and 'widths' in meta) else None
                out = model(signals)
                losses = criterion(out, labels, centers, widths)
                va_loss += float(losses['loss'].item())
            va_loss /= max(1, len(val_loader))
        scheduler.step(va_loss)

        print(f"Epoch {epoch}: train {tr_loss:.4f}  val {va_loss:.4f}  lr {optimizer.param_groups[0]['lr']:.3e}")

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': va_loss,
        }
        torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch:02d}.pth"))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(save_dir, "best.pth"))


# ------------------------------
# Minimal usage example (pseudocode for dataset)
# ------------------------------
"""
from defect_focused_dataset import get_defect_focused_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader = get_defect_focused_dataloader(
    root_path="json_data_0718",
    batch_size=8,
    seq_length=50,
    shuffle=True,
    validation_split=0.2,
    min_defects_per_sequence=1
)
# If your loader can also yield centers/widths, put them into meta['centers'], meta['widths'] normalized to [0,1].

model = Hybrid1D_DetLoc(signal_length=320).to(device)
train_detloc(model, train_loader, val_loader, device, save_dir='models/h1d_detloc')

# Inference on one batch (B=1):
model.eval()
with torch.no_grad():
    sample, _, _ = next(iter(val_loader))
    pred = model.infer_single(sample[:1].to(device))
    print(pred['prob'].shape, pred['center'].shape, pred['width'].shape)  # [N], [N], [N]
"""
