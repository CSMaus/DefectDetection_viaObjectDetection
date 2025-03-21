import os
from ultralytics import YOLO
# TODO: Upgrade to torch>=2.0.0 for deterministic training
def main():
    data_yaml = "data.yaml"
    model = YOLO("yolov5su.pt")
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,  # 320
        batch=4,
        name="my_yolov5_run",
        device=0
    )

if __name__ == "__main__":
    main()
