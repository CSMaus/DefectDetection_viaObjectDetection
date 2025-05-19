from transformers import AutoProcessor, AutoModelForObjectDetection
# was transformers-4.49.0
# needed to upgrade to use new models from hugging-face:
# installed huggingface-hub-0.31.2 transformers-4.51.3

processor = AutoProcessor.from_pretrained("ustc-community/dfine-nano-coco")
model = AutoModelForObjectDetection.from_pretrained("ustc-community/dfine-nano-coco")




