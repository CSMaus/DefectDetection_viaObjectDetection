
# yolo_data/
# ├─ images/
# │   └─ train/
# │       └─ (your .png, .jpg files)
# ├─ labels/
# │   └─ train/
# │       └─ (matching .txt files in YOLO format)
# └─ data.yaml
import os
import json
import cv2
import shutil

def prepare_yolo_data(
    dataset_folder="dataset",
    annotations_file="annotations.json",
    output_folder="yolo_data"
):
    images_out = os.path.join(output_folder, "images", "train")
    images_out_val = os.path.join(output_folder, "images", "val")
    labels_out = os.path.join(output_folder, "labels", "train")
    labels_out_val = os.path.join(output_folder, "labels", "val")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    os.makedirs(images_out_val, exist_ok=True)
    os.makedirs(labels_out_val, exist_ok=True)

    with open(annotations_file, "r") as f:
        annotations = json.load(f)

    for seq_name in list(annotations.keys())[:-2]:
        seq_path = os.path.join(dataset_folder, seq_name)
        if not os.path.isdir(seq_path):
            print(f"Warning: sequence folder not found: {seq_path}")
            continue

        ann_dict = annotations[seq_name]

        for img_name in ann_dict.keys():
            img_path = os.path.join(seq_path, img_name)
            if not os.path.isfile(img_path):
                print(f"Warning: image file not found: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: could not read image: {img_path}")
                continue
            h, w, _ = img.shape

            base_name = f"{seq_name}_{img_name}"
            base_name = base_name.replace("/", "_").replace("\\", "_")

            out_img_path = os.path.join(images_out, base_name)
            shutil.copy2(img_path, out_img_path)

            label_path = os.path.join(labels_out, os.path.splitext(base_name)[0] + ".txt")

            bboxes = ann_dict[img_name]  # list of dicts like {"bbox": [...], "label": "..."}
            if not bboxes:
                # If no bounding boxes, create empty txt
                with open(label_path, "w") as lf:
                    lf.write("")
                continue

            yolo_lines = []
            for obj in bboxes:
                box = obj["bbox"]
                x_min, x_max, y_min, y_max = box

                # YOLO expects [class, x_center, y_center, width, height] normalized to 0..1
                # class = 0 (only one class)
                class_id = 0

                box_w = (x_max - x_min)
                box_h = (y_max - y_min)
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0

                x_center_norm = x_center / float(w)
                y_center_norm = y_center / float(h)
                w_norm = box_w / float(w)
                h_norm = box_h / float(h)

                line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                yolo_lines.append(line)

            with open(label_path, "w") as lf:
                for y_line in yolo_lines:
                    lf.write(y_line + "\n")

    for seq_name in list(annotations.keys())[-2:]:
        seq_path = os.path.join(dataset_folder, seq_name)
        if not os.path.isdir(seq_path):
            print(f"Warning: sequence folder not found: {seq_path}")
            continue

        ann_dict = annotations[seq_name]

        for img_name in ann_dict.keys():
            img_path = os.path.join(seq_path, img_name)
            if not os.path.isfile(img_path):
                print(f"Warning: image file not found: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: could not read image: {img_path}")
                continue
            h, w, _ = img.shape

            base_name = f"{seq_name}_{img_name}"
            base_name = base_name.replace("/", "_").replace("\\", "_")

            out_img_path = os.path.join(images_out_val, base_name)
            shutil.copy2(img_path, out_img_path)

            label_path = os.path.join(labels_out_val, os.path.splitext(base_name)[0] + ".txt")

            bboxes = ann_dict[img_name]  # list of dicts like {"bbox": [...], "label": "..."}
            if not bboxes:
                # If no bounding boxes, create empty txt
                with open(label_path, "w") as lf:
                    lf.write("")
                continue

            yolo_lines = []
            for obj in bboxes:
                box = obj["bbox"]
                x_min, x_max, y_min, y_max = box

                # YOLO expects [class, x_center, y_center, width, height] normalized to 0..1
                # class = 0 (only one class)
                class_id = 0

                box_w = (x_max - x_min)
                box_h = (y_max - y_min)
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0

                x_center_norm = x_center / float(w)
                y_center_norm = y_center / float(h)
                w_norm = box_w / float(w)
                h_norm = box_h / float(h)

                line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                yolo_lines.append(line)

            with open(label_path, "w") as lf:
                for y_line in yolo_lines:
                    lf.write(y_line + "\n")

    print(f"YOLO data prepared in: {output_folder}")
    print("images/train -> images, labels/train -> YOLO labels")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    dataset_path = os.path.join(parent_dir, "dataset")
    annotation_file = os.path.join(parent_dir, "annotations.json")
    prepare_yolo_data(
        dataset_folder=dataset_path,
        annotations_file=annotation_file,
        output_folder="datasets/data0321"
    )


