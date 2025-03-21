import os
import json
import cv2
import shutil
import random

def prepare_yolo_data(
    dataset_folder="dataset",
    annotations_file="annotations.json",
    output_folder="yolo_data",
    no_bbox_keep_ratio=0.1,  # e.g., keep 10% of images that have no bboxes
    val_ratio = 0.15
):
    images_out = os.path.join(output_folder, "images", "train")
    images_out_val = os.path.join(output_folder, "images", "val")
    labels_out = os.path.join(output_folder, "labels", "train")
    labels_out_val = os.path.join(output_folder, "labels", "val")
    if not os.path.exists(images_out):
        os.makedirs(images_out)
        os.makedirs(labels_out)
        os.makedirs(images_out_val)
        os.makedirs(labels_out_val)

    with open(annotations_file, "r") as f:
        annotations = json.load(f)

    images_with_bboxes = []
    images_no_bboxes = []

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
                images_with_bboxes.append((seq_name, img_name, bboxes))
            else:
                images_no_bboxes.append((seq_name, img_name, bboxes))

    random.shuffle(images_no_bboxes)
    # keep_count = int(len(images_no_bboxes) * no_bbox_keep_ratio)
    # images_no_bboxes = images_no_bboxes[:keep_count]
    if val_ratio != 0:
        random.shuffle(images_with_bboxes)
        val_count = int(len(images_with_bboxes) * 0.15)  # 15% for validation
        val_images = images_with_bboxes[:val_count]
        train_images = images_with_bboxes[val_count:]
        for dataset, image_set in [('train', train_images), ('val', val_images)]:
            images_dir = images_out if dataset == 'train' else images_out_val
            labels_dir = labels_out if dataset == 'train' else labels_out_val

            for seq_name, img_name, bboxes in image_set:
                seq_path = os.path.join(dataset_folder, seq_name)
                img_path = os.path.join(seq_path, img_name)

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: could not read image: {img_path}")
                    continue

                h, w, _ = img.shape
                base_name = f"{seq_name}_{img_name}"
                base_name = base_name.replace("/", "_").replace("\\", "_")

                out_img_path = os.path.join(images_dir, base_name)
                shutil.copy2(img_path, out_img_path)

                label_path = os.path.join(labels_dir, os.path.splitext(base_name)[0] + ".txt")

                yolo_lines = []
                for obj in bboxes:
                    box = obj["bbox"]
                    x_min, x_max, y_min, y_max = box

                    class_id = 0
                    box_w = abs(x_max - x_min)
                    box_h = abs(y_max - y_min)
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
    else:
        all_images = images_with_bboxes # + images_no_bboxes
        for seq_name, img_name, bboxes in all_images:
            seq_path = os.path.join(dataset_folder, seq_name)
            img_path = os.path.join(seq_path, img_name)

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

            if not bboxes:
                with open(label_path, "w") as lf:
                    lf.write("")
                continue

            yolo_lines = []
            for obj in bboxes:
                box = obj["bbox"]  # [x_min, x_max, y_min, y_max]
                x_min, x_max, y_min, y_max = box

                class_id = 0  # single class
                box_w = abs(x_max - x_min)
                box_h = abs(y_max - y_min)
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
    # print(f"Kept {len(images_with_bboxes)} images with bboxes + {keep_count} no-bbox images.")
    print("images/train -> your images, labels/train -> YOLO labels")


if __name__ == "__main__":
    prepare_yolo_data(
        dataset_folder="dataset",
        annotations_file="annotations.json",
        output_folder="yolo/datasets/data0321",
        no_bbox_keep_ratio=0,
        val_ratio=0.15,
    )
