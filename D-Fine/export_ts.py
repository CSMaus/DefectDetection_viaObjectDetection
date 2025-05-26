# predict_sequence.py
import sys, json, torch, numpy as np
from PIL import Image
from types import SimpleNamespace
from temporal_dfine import TemporalDFine

def main():
    # 1) load sequence from stdin
    data = json.load(sys.stdin)   # data is List[List[List[float]]] shape [T][H][W]
    images = []
    for arr in data:
        a = np.array(arr, dtype=np.float32)
        # assume pixel range [0,1] or [0,255]
        if a.max() <= 1.0:
            a = (a * 255).astype(np.uint8)
        else:
            a = a.astype(np.uint8)
        img = Image.fromarray(a)
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    # 2) load model
    ckpt = sys.argv[1]
    device = torch.device("cpu")
    model = TemporalDFine().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 3) preprocess & inference
    processor = model.processor
    sizes = [im.size[::-1] for im in images]  # (H, W)
    proc = processor(images=images, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)

    with torch.no_grad():
        results = model(pixel_values, sizes, targets=None)

    # 4) format output
    out = []
    for res in results:
        frame = []
        for box, lab, sc in zip(res["boxes"].tolist(),
                                res["labels"].tolist(),
                                res["scores"].tolist()):
            frame.append({
                "box": box,
                "label": model.dfine.config.id2label[lab],
                "score": sc
            })
        out.append(frame)

    json.dump(out, sys.stdout)

if __name__ == "__main__":
    main()