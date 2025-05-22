# here I want to see the output of NN from feature extraction layer
# and what input should be placed to the classification and object detection module
import sys

import torch
import requests
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from types import SimpleNamespace
from transformers.models.d_fine.modeling_d_fine import weighting_function

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
print("Original image size is:", image.size)

image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-small-coco", use_fast=True)
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-small-coco")

inputs = image_processor(images=image, return_tensors="pt")

'''
with torch.no_grad():
    outputs = model.model(**inputs)

# Get feature map before classification and bbox heads - THIS IS CORRECT!!
features = outputs.last_hidden_state  # shape: (batch_size, num_queries, hidden_dim)
print("Feature shape:", features.shape)
'''
with (torch.no_grad()):
    outputs = model.model(**inputs)  # model.model.encoder(inputs["pixel_values"])
    # outputs = model.model(inputs["pixel_values"])
    features = outputs.last_hidden_state
    # init_ref = outputs.initial_reference_points  # (B, layers, Nq, 4)
    # init_ref = outputs.init_reference_points  # (B, Nq, 4)
    print("Feature shape:", features.shape)


'''with torch.no_grad():
    class_logits = model.class_embed[-1](features)  # shape: (B, N_queries, num_classes)
    bbox_dist = model.bbox_embed[-1](features)   # shape: (B, N, 4*(max_num_bins+1))
    print("BBox dist shape ", bbox_dist.shape)
    # bbox_deltas = model.bbox_embed[-1](features)[..., :4]

    # build the W(n) “project” vector exactly as the decoder does
    project = weighting_function(
        model.config.max_num_bins,
        model.model.decoder.up,
        model.model.decoder.reg_scale,
    )
    # decode the 4-bin distributions into real [cx, cy, w, h] coords
    bbox_deltas_real = model.model.decoder.integral(bbox_dist, project) + init_ref  # (B, Nq, 4)

'''
class_logits = outputs.intermediate_logits[:, -1]             # (B, num_queries, num_classes)
pred_boxes   = outputs.intermediate_reference_points[:, -1]  # (B, num_queries, 4)

# torch.save(features, "extracted_features.pt")
print("Class logits shape:", class_logits.shape)
print("BBox deltas shape:", bbox_deltas_real.shape)

processed_real = image_processor.post_process_object_detection(
    SimpleNamespace(logits=class_logits, pred_boxes=bbox_deltas_real),
    target_sizes=torch.tensor([image.size[::-1]]),
    threshold=0.3
)

print("Image size target", image.size[::-1])

plt.figure(figsize=(12, 8))
plt.imshow(image)
ax = plt.gca()

for score, label_id, box in zip(processed_real[0]["scores"], processed_real[0]["labels"], processed_real[0]["boxes"]):
    if score > 0.5:
        label = model.config.id2label[label_id.item()]
        score_val = score.item()
        x, y, x2, y2 = box.tolist()
        w = x2 - x
        h = y2 - y
        print(f"{label}: score = {score:.2f}, x,y,x2,y2: {x:.2f}, {y:.2f}, {x2:.2f}, {y2:.2f}")

        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f"{label}: {score_val:.2f}", fontsize=10, color='white', backgroundcolor='green')
plt.title("Real Features")
plt.axis("off")
plt.tight_layout()
plt.show()




sys.exit()
dummy_features = torch.randn(1, 300, model.config.hidden_size)

with torch.no_grad():
    class_logits_d = model.class_embed[-1](dummy_features)
    bbox_deltas_d = model.bbox_embed[-1](dummy_features)[..., :4]
print("For dummy features:")
print("Class logits shape:", class_logits_d.shape)
print("BBox deltas shape:", bbox_deltas_d.shape)

bbox_deltas_dummy = bbox_deltas_d

processed_dummy = image_processor.post_process_object_detection(
    SimpleNamespace(logits=class_logits_d, pred_boxes=bbox_deltas_dummy),
    target_sizes=torch.tensor([image.size[::-1]]),
    threshold=0.3
)
plt.figure(figsize=(12, 8))
plt.imshow(image)
ax = plt.gca()
for score, label_id, box in zip(processed_dummy[0]["scores"], processed_dummy[0]["labels"], processed_dummy[0]["boxes"]):
    if score > 0.5:
        label = model.config.id2label[label_id.item()]
        score_val = score.item()
        x, y, x2, y2 = box.tolist()
        rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f"{label}: {score_val:.2f}", fontsize=10, color='white', backgroundcolor='red')
plt.title("Dummy Features")
plt.axis("off")
plt.tight_layout()
plt.show()


