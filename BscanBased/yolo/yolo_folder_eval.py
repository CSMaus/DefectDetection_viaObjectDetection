import os
import cv2
from ultralytics import YOLO


def main(folder_full_path, to_save_path):
    # project = "C:/Users/oem/Desktop/Kseniia/BscanBased/yolov5/"
    project = "yolo11n_0321/"
    name = "run/weights/best.pt"
    model_path = os.path.join(project, name)  # project,"name.pt")
    model = YOLO(model_path)

    for img_name in os.listdir(folder_full_path):
        img_path = os.path.join(folder_full_path, img_name)
        results = model.predict(img_path)
        res = results[0]

        boxes = res.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy.tolist()[0]  # correct indexing
            conf = float(box.conf)
            cls_id = int(box.cls)
            print(f"Box {i}: xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={conf:.2f}, class={cls_id}")

        plotted = res.plot()
        img_path_tosave = os.path.join(to_save_path, f'{img_name}.png')
        cv2.imwrite(img_path_tosave, plotted)




if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    to_save_pred_path = os.path.join(parent_dir, "predictions/787-404_07_Ch-0/")
    # folder where images for prediction are stored
    folder_path = os.path.join(parent_dir, "dataset/787-404_07_Ch-0/")

    if not os.path.exists(to_save_pred_path):
        os.makedirs(to_save_pred_path)

    if not os.path.exists(folder_path):
        print(f"Error: folder {folder_path} not found.")
    main(folder_path, to_save_pred_path)



