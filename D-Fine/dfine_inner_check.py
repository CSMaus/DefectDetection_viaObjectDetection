# here I want to see the output of NN from feature extraction layer
# and what input should be placed to the classification and object detection module
import torch
import requests
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from types import SimpleNamespace

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-small-coco", use_fast=True)
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-small-coco")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model.model(**inputs)

# Get feature map before classification and bbox heads
features = outputs.last_hidden_state  # shape: (batch_size, num_queries, hidden_dim)
print("Feature shape:", features.shape)

dummy_features = torch.randn(1, 300, model.config.hidden_size)

with torch.no_grad():
    class_logits = model.class_embed[-1](features)  # shape: (batch_size, num_queries, num_classes)
    bbox_deltas = model.bbox_embed[-1](features)[..., :4]
bbox_deltas_real = bbox_deltas
# torch.save(features, "extracted_features.pt")
print("Class logits shape:", class_logits.shape)
print("BBox deltas shape:", bbox_deltas.shape)

with torch.no_grad():
    class_logits_d = model.class_embed[-1](dummy_features)
    bbox_deltas_d = model.bbox_embed[-1](dummy_features)[..., :4]
print("For dummy features:")
print("Class logits shape:", class_logits_d.shape)
print("BBox deltas shape:", bbox_deltas_d.shape)

bbox_deltas_dummy = bbox_deltas_d
processed_real = image_processor.post_process_object_detection(
    SimpleNamespace(logits=class_logits, pred_boxes=bbox_deltas_real),
    target_sizes=torch.tensor([image.size[::-1]]),
    threshold=0.3
)
processed_dummy = image_processor.post_process_object_detection(
    SimpleNamespace(logits=class_logits_d, pred_boxes=bbox_deltas_dummy),
    target_sizes=torch.tensor([image.size[::-1]]),
    threshold=0.3
)
plt.figure(figsize=(12, 8))
plt.imshow(image)
ax = plt.gca()
for score, label_id, box in zip(processed_real[0]["scores"], processed_real[0]["labels"], processed_real[0]["boxes"]):
    if score > 0.5:
        label = model.config.id2label[label_id.item()]
        score_val = score.item()
        x, y, x2, y2 = box.tolist()
        rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f"{label}: {score_val:.2f}", fontsize=10, color='white', backgroundcolor='green')
plt.title("Real Features")
plt.axis("off")
plt.tight_layout()
plt.show()


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


