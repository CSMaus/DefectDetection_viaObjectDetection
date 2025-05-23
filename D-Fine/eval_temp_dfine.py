# code for predictions
import os
import torch
from types import SimpleNamespace
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor
from temporal_dfine import TemporalDFine
from transformers.models.d_fine.modeling_d_fine import distance2bbox

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

    # 1) load your trained TemporalDFine
    model = TemporalDFine().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    processor = model.processor
    id2label  = model.dfine.config.id2label

    # 2) collect & sort images
    img_paths = [
        os.path.join(input_folder, fn)
        for fn in os.listdir(input_folder)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    img_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    os.makedirs(output_folder, exist_ok=True)

    # 3) process in chunks of seq_len
    for chunk in chunk_list(img_paths, seq_len):
        imgs  = [Image.open(p).convert("RGB") for p in chunk]
        sizes = [img.size[::-1] for img in imgs]  # list of (H, W)

        # tokenize
        proc = processor(images=imgs, return_tensors="pt").to(device)
        pixel_values = proc["pixel_values"]  # (T,3,H,W)

        with torch.no_grad():
            # a) backbone → features & reference points
            out      = model.dfine.model(pixel_values=pixel_values)
            feats    = out.last_hidden_state           # (T, Q, D)
            init_ref = out.init_reference_points       # (T, Q, 4)

            # b) fuse temporally
            fused = model.temporal_encoder(feats)      # (T, Q, D)

            # c) heads → logits & raw distributions
            cls_logits = model.class_head(fused)       # (T, Q, C+1)
            bbox_dist  = model.bbox_head(fused)        # (T, Q, 4*(bins+1))

            # d) build the “projection” vector Wₙ yourself:
            BINS      = model.max_num_bins
            up        = model.up
            reg_scale = model.reg_scale
            # make it a 1‐D float tensor on the right device & dtype:
            Wn = (
                torch.arange(BINS + 1, device=bbox_dist.device, dtype=bbox_dist.dtype)
                * up
                / reg_scale
            )

            # e) convert distributions → distances → boxes
            distances = model.integral(bbox_dist, Wn)                             # (T, Q, 4)
            bbox_pred = distance2bbox(init_ref, distances, reg_scale).clamp(0, 1)  # (T, Q, 4)

        # 4) post‐process each frame & save
        for i, path in enumerate(chunk):
            outs = SimpleNamespace(
                logits     = cls_logits[i : i + 1],
                pred_boxes = bbox_pred[i : i + 1],
            )
            res = processor.post_process_object_detection(
                outs,
                target_sizes=torch.tensor([sizes[i]], device=device),
                threshold=0.3,
            )[0]

            img    = imgs[i]
            boxes  = res["boxes"].cpu().numpy()
            labels = res["labels"].cpu().numpy()
            scores = res["scores"].cpu().numpy()

            img = draw_boxes(img, boxes, labels, scores, id2label)
            out_path = os.path.join(output_folder, os.path.basename(path))
            img.save(out_path)
            print("Saved:", out_path)

    print("Done. All predictions in", output_folder)


if __name__ == "__main__":
    # FOLDER = "WOT-D456_A4_002_Ch-0_D0.5-12"
    FOLDER = "WOT D44-D47_01_Ch-0_D3-7"
    # INPUT_FOLDER = os.path.join("ds_manipulations/dataset/WOT-20250522(auto)", FOLDER)
    INPUT_FOLDER = os.path.join("ds_manipulations", FOLDER)
    OUTPUT_FOLDER = os.path.join("ds_manipulations/dataset/predictions", FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    CKPT = "temporal_dfine_epoch_2.pth"
    SEQ_LEN = 50

    inference_on_folder(INPUT_FOLDER, OUTPUT_FOLDER, CKPT, SEQ_LEN)



