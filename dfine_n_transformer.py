# Draft for future with transformers

import torch.nn as nn
import torch
import requests
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from types import SimpleNamespace
from transformers.models.d_fine.modeling_d_fine import weighting_function
from transformers.models.d_fine.modeling_d_fine import distance2bbox

image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-small-coco", use_fast=True)
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-small-coco")


# 1) build a small TransformerEncoder that preserves (B, Q, D) shape
encoder_layer = nn.TransformerEncoderLayer(
    d_model=model.config.hidden_size,   # 256
    nhead=8,                            # you can tune
    dim_feedforward=1024,
    dropout=0.1,
    batch_first=True                    # so input/output are (B, Q, D)
)
temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

# 2) suppose you have already stacked your N images’ features into:
#    `all_feats` of shape (B, num_queries, hidden_dim)
#    e.g. by looping or via a batched forward of your backbone+encoder:
#       all_feats = torch.cat(list_of_per_image_features, dim=0)

# apply your new Transformer over the *query* dimension:
transformed_feats = temporal_transformer(all_feats)
# -> still (B, num_queries, hidden_dim)

# 3) now just feed each row back into your head exactly as before:
class_logits = model.class_embed[-1](transformed_feats)   # (B, Q, C)
bbox_dist   = model.bbox_embed[-1](transformed_feats)     # (B, Q, 4*(bins+1))
project     = weighting_function(
    model.config.max_num_bins,
    model.model.decoder.up,
    model.model.decoder.reg_scale,
)
distances   = model.model.decoder.integral(bbox_dist, project)  # (B, Q, 4)
init_ref    = outputs.initial_reference_points[:, 0]            # (B, Q, 4)
bbox_pred   = distance2bbox(init_ref, distances, model.config.reg_scale)

# 4) now do your post‐processing per‐image as usual…
for i in range(B):
    res = image_processor.post_process_object_detection(
       SimpleNamespace(logits=class_logits[i:i+1], pred_boxes=bbox_pred[i:i+1]),
       target_sizes=torch.tensor([sizes[i]]),   # keep track of each image’s (h,w)
       threshold=0.3,
    )
    # …draw res[0] on image_i…
