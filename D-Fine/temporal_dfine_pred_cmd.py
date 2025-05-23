# code for predictions

import os
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor
from temporal_dfine import TemporalDFine


def chunk_list(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def draw_boxes(image: Image.Image, boxes, labels, scores, id2label):
    """Draws bounding boxes and labels onto a PIL image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for box, lbl, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{id2label[int(lbl)]}: {sc:.2f}"
        draw.text((x1, y1), text, fill="red", font=font)
    return image


def main():
    parser = argparse.ArgumentParser(description="Run TemporalDFine on image sequences and save predictions.")
    parser.add_argument("--input_folder", required=True, help="Path to folder containing image sequence")
    parser.add_argument("--output_folder", required=True, help="Directory to save images with drawn predictions")
    parser.add_argument("--model_checkpoint", default="temporal_dfine_final.pth", help="Path to trained TemporalDFine weights (.pth)")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length for temporal model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = TemporalDFine(checkpoint="ustc-community/dfine-small-coco")
    state = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    processor = model.processor
    id2label  = model.dfine.config.id2label

    # collect and sort image paths
    img_paths = [os.path.join(args.input_folder, fn)
                 for fn in os.listdir(args.input_folder)
                 if fn.lower().endswith((".png", ".jpg", ".jpeg"))]
    img_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    os.makedirs(args.output_folder, exist_ok=True)

    # process in chunks
    for chunk in chunk_list(img_paths, args.seq_len):
        images = [Image.open(p).convert("RGB") for p in chunk]
        sizes  = [img.size[::-1] for img in images]  # (H, W)

        # prepare model inputs
        proc = processor(images=images, return_tensors="pt").to(device)
        pixel_values = proc["pixel_values"]

        # forward pass
        with torch.no_grad():
            results = model(pixel_values, sizes, targets=None)

        # draw and save each image
        for path, res in zip(chunk, results):
            img = Image.open(path).convert("RGB")
            boxes  = res["boxes"].cpu().numpy()
            labels = res["labels"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            img = draw_boxes(img, boxes, labels, scores, id2label)
            out_path = os.path.join(args.output_folder, os.path.basename(path))
            img.save(out_path)
            print(f"Saved: {out_path}")

    print("All sequences processed. Predictions saved to:", args.output_folder)


if __name__ == "__main__":
    main()



