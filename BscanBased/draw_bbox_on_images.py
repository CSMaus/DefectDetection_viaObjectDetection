import os
import json
import cv2

def save_images_with_bboxes(sequence_folder, annotations_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    sequence_name = os.path.basename(os.path.normpath(sequence_folder))
    sequence_annotations = annotations.get(sequence_name, {})

    for img_name, defects in sequence_annotations.items():
        if not defects:
            continue

        img_path = os.path.join(sequence_folder, img_name)
        if not os.path.isfile(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        height, width = img.shape[:2]

        for defect in defects:
            x_min, x_max, y_min_norm, y_max_norm = defect["bbox"]

            # Convert normalized y-coordinates to absolute pixel indices
            y_min = y_min_norm  # int(y_min_norm * height)
            y_max = y_max_norm  # int(y_max_norm * height)

            # x coordinates are already pixel indices
            x_min = int(x_min)
            x_max = int(x_max)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(img, defect["label"], (x_max, max(0, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        output_path = os.path.join(output_folder, f"bbox_{img_name}")
        cv2.imwrite(output_path, img)
        # break
    print(f"Images with bounding boxes saved to: {output_folder}")

if __name__ == "__main__":
    sequence_folder = "dataset/787-225_01_Ch-0/"
    annotations_path = "annotations.json"
    output_folder = "data_with_bbox/787-225_01_Ch-0/"

    save_images_with_bboxes(sequence_folder, annotations_path, output_folder)
