import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DFineForObjectDetection, AutoImageProcessor
from transformers.models.d_fine.modeling_d_fine import weighting_function, distance2bbox

# ─── Temporal‐DFine wrapper ───────────────────────────────────────────────

class TemporalDFine(nn.Module):
    def __init__(self, checkpoint="ustc-community/dfine-small-coco"):
        super().__init__()
        # 1) load the full DFine detection model and its processor
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
        self.dfine    = DFineForObjectDetection.from_pretrained(checkpoint)

        # 2) freeze all DFine weights (we’ll train only our transformer)
        for p in self.dfine.parameters():
            p.requires_grad = False

        # 3) build a small TransformerEncoder to fuse features across frames
        d_model = self.dfine.config.hidden_size  # 256
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 4) shortcut references to the DFine heads + integral logic
        self.class_head  = self.dfine.class_embed[-1]
        self.bbox_head   = self.dfine.bbox_embed[-1]
        self.integral    = self.dfine.model.decoder.integral
        self.up          = self.dfine.model.decoder.up
        self.reg_scale   = self.dfine.config.reg_scale
        self.max_num_bins= self.dfine.config.max_num_bins

    def forward(self, pixel_values, sizes):
        # pixel_values: (B, 3, H, W), sizes: list of (H, W) per image
        # 1) run the DFine backbone+encoder+decoder to get features
        out: torch.Tensor = self.dfine.model(pixel_values=pixel_values)
        feats           = out.last_hidden_state                  # (B, Q, D)
        init_ref        = out.init_reference_points              # (B, Q, 4)

        # 2) fuse them across the batch dimension as "temporal" frames
        fused = self.temporal_encoder(feats)                    # (B, Q, D)

        # 3) classification & bbox‐distribution heads
        cls_logits = self.class_head(fused)                     # (B, Q, num_classes)
        bbox_dist  = self.bbox_head(fused)                      # (B, Q, 4*(bins+1))

        # 4) decode distributions → center/x-y/w-h
        proj       = weighting_function(
                        self.max_num_bins,
                        self.up,
                        self.dfine.model.decoder.reg_scale
                    )
        distances  = self.integral(bbox_dist, proj)            # (B, Q, 4)
        bbox_pred  = distance2bbox(init_ref, distances, self.reg_scale)

        # 5) convert to pixel‐space boxes + scores per image
        results = []
        for i, size in enumerate(sizes):
            single = pixel_values[i : i+1]  # keep alive for processor
            out_i  = self.processor.post_process_object_detection(
                         {"logits": cls_logits[i:i+1], "pred_boxes": bbox_pred[i:i+1]},
                         target_sizes=torch.tensor([size]),
                         threshold=0.3,
                     )
            results.append(out_i[0])
        return results


# ─── Placeholder DataLoader & Training Loop ─────────────────────────────────

# fill in dataset later to return:
#   batch = {"pixel_values": Tensor(B,3,H,W), "sizes": [(H,W),...], "targets": ...}
train_loader = DataLoader(...)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TemporalDFine().to(device)
opt    = torch.optim.Adam(model.temporal_encoder.parameters(), lr=1e-4)  # only transformer trains

for epoch in range(10):
    for batch in train_loader:
        pv    = batch["pixel_values"].to(device)
        sizes = batch["sizes"]
        # targets = batch["targets"]  # if you want to compute loss

        # forward
        outputs = model(pv, sizes)

        # TODO: compute your detection loss against `batch["targets"]`
        # loss = detection_loss(outputs, batch["targets"])
        # opt.zero_grad(); loss.backward(); opt.step()

        # or for pure inference:
        # for img_i, res_i in zip(batch["images"], outputs):
        #     draw_boxes_on_image(img_i, res_i["boxes"], res_i["scores"], res_i["labels"])











