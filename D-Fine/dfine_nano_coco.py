import torch
import requests
from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-small-coco", use_fast=True)
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-small-coco")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

drawn_image = image.copy()
plt.figure(figsize=(12, 8))
plt.imshow(drawn_image)
ax = plt.gca()
for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        label = model.config.id2label[label_id.item()]
        score_val = score.item()
        x, y, w, h = box.tolist()
        rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f"{label}: {score_val:.2f}", fontsize=10, color='white', backgroundcolor='red')

plt.axis("off")
plt.tight_layout()
plt.show()