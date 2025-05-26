# code for predictions
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from transformers import (
    DFineForObjectDetection,
    AutoImageProcessor,
    DFineConfig,
)
from transformers.models.d_fine.modeling_d_fine import (
    weighting_function,
    distance2bbox,
)

# Import the TemporalDFine class from temp_dfine_over.py
from temp_dfine_over import TemporalDFine

def chunk_list(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def draw_boxes(image: Image.Image, boxes, labels, scores, id2label):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    for (x1, y1, x2, y2), lbl, sc in zip(boxes, labels, scores):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{id2label[int(lbl)]}: {sc:.2f}"
        draw.text((x1, y1), text, fill="red", font=font)
    return image

def inference_on_folder(input_folder, output_folder, checkpoint, seq_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the checkpoint to get the model configuration
    checkpoint_data = torch.load(checkpoint, map_location=device)
    
    # Check if the checkpoint contains the model state dict directly or has metadata
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
        # Try to extract id2label from metadata if available
        id2label = checkpoint_data.get("id2label", None)
        num_defect_classes = checkpoint_data.get("num_defect_classes", None)
    else:
        state_dict = checkpoint_data
        id2label = None
        num_defect_classes = None
    
    # If we couldn't extract id2label from the checkpoint, we need to recreate it
    if id2label is None:
        # Try to determine the number of classes from the state dict
        for key, value in state_dict.items():
            if "class_head.weight" in key:
                # The shape is [num_classes, hidden_dim]
                num_classes = value.shape[0]
                num_defect_classes = num_classes - 1  # Subtract 1 for the "no_object" class
                print(f"Detected {num_defect_classes} defect classes from checkpoint")
                break
        
        # Create a simple id2label mapping
        id2label = {i: f"defect_{i}" for i in range(num_defect_classes)}
        id2label[num_defect_classes] = "no_object"
    
    # 1) Initialize the model with the same configuration as during training
    model = TemporalDFine(
        checkpoint="ustc-community/dfine-small-coco",
        num_defect_classes=num_defect_classes,
        id2label=id2label,
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {checkpoint}")

    processor = model.processor
    id2label = model.dfine.config.id2label
    print(f"Classes: {id2label}")

    # 2) collect & sort images
    img_paths = [
        os.path.join(input_folder, fn)
        for fn in os.listdir(input_folder)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    img_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    print(f"Found {len(img_paths)} images in {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    # 3) process in chunks of seq_len
    for i, chunk in enumerate(chunk_list(img_paths, seq_len)):
        print(f"Processing chunk {i+1}/{(len(img_paths) + seq_len - 1) // seq_len}")
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        sizes = [img.size[::-1] for img in imgs]  # list of (H, W)

        # tokenize
        proc = processor(images=imgs, return_tensors="pt").to(device)
        pixel_values = proc["pixel_values"]  # (T,3,H,W)

        with torch.no_grad():
            # a) backbone → features & reference points
            out = model.dfine.model(pixel_values=pixel_values)
            feats = out.last_hidden_state           # (T, Q, D)
            init_ref = out.init_reference_points    # (T, Q, 4)

            # b) fuse temporally
            fused = model.temporal_encoder(feats)   # (T, Q, D)

            # c) heads → logits & raw distributions
            cls_logits = model.class_head(fused)    # (T, Q, C+1)
            bbox_dist = model.bbox_head(fused)      # (T, Q, 4*(bins+1))
            
            # Handle potential NaN values
            cls_logits = torch.nan_to_num(
                cls_logits,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
            
            bbox_dist = torch.nan_to_num(
                bbox_dist,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )

            # d) build the "projection" vector Wₙ yourself:
            Wn = torch.arange(
                model.max_num_bins + 1,
                device=bbox_dist.device,
                dtype=bbox_dist.dtype,
            ) * model.up / model.reg_scale

            # e) convert distributions → distances → boxes
            distances = model.integral(bbox_dist, Wn)                             # (T, Q, 4)
            distances = torch.nan_to_num(distances, nan=0.0, posinf=1.0, neginf=0.0)
            bbox_pred = distance2bbox(init_ref, distances, model.reg_scale).clamp(0, 1)  # (T, Q, 4)
            bbox_pred = torch.nan_to_num(bbox_pred, nan=0.5, posinf=1.0, neginf=0.0)

        # 4) post‐process each frame & save
        for i, path in enumerate(chunk):
            outs = SimpleNamespace(
                logits=cls_logits[i : i + 1],
                pred_boxes=bbox_pred[i : i + 1],
            )
            res = processor.post_process_object_detection(
                outs,
                target_sizes=torch.tensor([sizes[i]], device=device),
                threshold=0.0,
            )[0]

            img = imgs[i]
            boxes = res["boxes"].cpu().numpy()
            labels = res["labels"].cpu().numpy()
            scores = res["scores"].cpu().numpy()

            img = draw_boxes(img, boxes, labels, scores, id2label)
            out_path = os.path.join(output_folder, os.path.basename(path))
            img.save(out_path)
            print(f"Saved: {out_path}")

    print(f"Done. All predictions saved in {output_folder}")


if __name__ == "__main__":
    # FOLDER = "D25-28_A8_02_Ch-0_D-2-20"
    FOLDER = "D25-28_A8_04_Ch-0_D-1.7-11"
    # INPUT_FOLDER = os.path.join("ds_manipulations/dataset/WOT-20250522(auto)", FOLDER)
    INPUT_FOLDER = os.path.join("ds_manipulations/", FOLDER)
    OUTPUT_FOLDER = os.path.join("ds_manipulations/dataset/predictions", FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Path to your trained model checkpoint
    CKPT = "temporal_dfine_epoch_0.pth"
    SEQ_LEN = 50

    inference_on_folder(INPUT_FOLDER, OUTPUT_FOLDER, CKPT, SEQ_LEN)
