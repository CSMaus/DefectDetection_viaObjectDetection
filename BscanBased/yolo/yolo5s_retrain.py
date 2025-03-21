from ultralytics import YOLO
# TODO: Upgrade to torch>=2.0.0 for deterministic training
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # this is bcs too many people works on this computer

def main():
    data_yaml = "data.yaml"
    model = YOLO("yolov5su.pt")
    model.train(
        data=data_yaml,
        epochs=30,
        imgsz=320,  # 640
        batch=4,
        name="bscan_yolov5_run3",
        project="C:/Users/oem/Desktop/Kseniia/BscanBased/yolo/",
        device=0
    )

if __name__ == "__main__":
    main()
