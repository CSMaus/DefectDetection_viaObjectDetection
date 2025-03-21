import os
import json
import cv2
import shutil
import random
import numpy as np


def prepare_yolo_seg_data(
        dataset_folder="dataset",
        annotations_file="annotations.json",
        output_folder="yolo_seg_data",
        no_mask_keep_ratio=0.1
):
    """
    Creates a YOLOv8 segmentation dataset:
      - images/ (RGB images)
      - labels/ (mask .png files matching each image)

    For each defect bounding box, we fill a rectangle area with 255 in the mask.
    All images with no bounding boxes: keep only a fraction (no_mask_keep_ratio).
    """

    images_out = os.path.join(output_folder, "images", "train")
    labels_out = os.path.join(output_folder, "labels", "train")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    with open(annotations_file, "r") as f:
        annotations = json.load(f)

    images_with_masks = []
    images_no_masks = []

    for seq_name, ann_dict in annotations.items():
        seq_path = os.path.join(dataset_folder, seq_name)
        if not os.path.isdir(seq_path):
            print(f"Warning: sequence folder not found: {seq_path}")
            continue

        for img_name, bboxes in ann_dict.items():
            img_path = os.path.join(seq_path, img_name)
            if not os.path.isfile(img_path):
                print(f"Warning: image file not found: {img_path}")
                continue

            if bboxes:
                images_with_masks.append((seq_name, img_name, bboxes))
            else:
                images_no_masks.append((seq_name, img_name, bboxes))

    # Keep only a fraction of the "no mask" images
    random.shuffle(images_no_masks)
    keep_count = int(len(images_no_masks) * no_mask_keep_ratio)
    images_no_masks = images_no_masks[:keep_count]

    all_images = images_with_masks + images_no_masks

    for seq_name, img_name, bboxes in all_images:
        seq_path = os.path.join(dataset_folder, seq_name)
        img_path = os.path.join(seq_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image: {img_path}")
            continue

        h, w, _ = img.shape

        # Copy image to yolo_seg_data/images/train
        base_name = f"{seq_name}_{img_name}"
        base_name = base_name.replace("/", "_").replace("\\", "_")
        out_img_path = os.path.join(images_out, base_name)
        shutil.copy2(img_path, out_img_path)

        # Create a single-channel mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # For each bounding box, fill that rectangle with 255
        for obj in bboxes:
            box = obj["bbox"]  # [x_min, x_max, y_min, y_max]
            x_min, x_max, y_min, y_max = map(int, box)

            # Clip to valid image range just in case
            x_min = max(0, min(x_min, w - 1))
            x_max = max(0, min(x_max, w - 1))
            y_min = max(0, min(y_min, h - 1))
            y_max = max(0, min(y_max, h - 1))

            # Fill rectangle on the mask
            mask[y_min:y_max, x_min:x_max] = 255

        # Save mask as .png next to image
        mask_name = os.path.splitext(base_name)[0] + ".png"
        out_mask_path = os.path.join(labels_out, mask_name)
        cv2.imwrite(out_mask_path, mask)

    print(f"Segmentation data prepared in: {output_folder}")
    print(f"Kept {len(images_with_masks)} images with defects + {keep_count} no-defect images.")
    print("images/train -> your images, labels/train -> your single-channel defect masks")


if __name__ == "__main__":
    prepare_yolo_seg_data(
        dataset_folder="dataset",
        annotations_file="annotations.json",
        output_folder="yolo_seg_data",
        no_mask_keep_ratio=0.1
    )
