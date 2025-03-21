import os
from ultralytics import YOLO
import cv2

def main():
    # project = "C:/Users/oem/Desktop/Kseniia/BscanBased/yolov5/"
    project = "yolo11n_0321/"
    name = "run/weights/best.pt"
    model_path = os.path.join(project, name)  # project,"name.pt")
    model = YOLO(model_path)
    img_name = "55.png"
    # img_name = "787-225_02_Ch-0_107.png"
    # img_name = "787-225_01_Ch-0_102.png"
    # img_name = "787-226_03_Ch-0_81.png"
    # img_name = "787-225_01_Ch-0_51.png"
    # img_name = "787-225_01_Ch-0_95.png"
    results = model.predict(img_name)
    res = results[0]

    '''boxes = res.boxes
    for i, box in enumerate(boxes):
        # box.xyxy -> tensor([ [x1, y1, x2, y2] ])
        # box.conf -> tensor([conf])
        # box.cls  -> tensor([class_id])
        x1, y1, x2, y2 = box.xyxy[0].tolist()   # Convert to Python float
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        print(f"Box {i}: xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={conf:.2f}, class={cls_id}")
    '''
    boxes = res.boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy.tolist()[0]  # correct indexing
        conf = float(box.conf)
        cls_id = int(box.cls)
        print(f"Box {i}: xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={conf:.2f}, class={cls_id}")

    plotted = res.plot()
    cv2.imshow("Predictions", plotted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(results)

if __name__ == "__main__":
    main()
