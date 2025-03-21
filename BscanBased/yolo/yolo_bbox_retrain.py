from ultralytics import YOLO
# TODO: Upgrade to torch>=2.0.0 for deterministic training
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # this is bcs too many people works on this computer

def main():
    data_yaml = "data.yaml"
    model = YOLO("yolo11n.yaml")
    model = YOLO("yolo11n.pt")
    model.train(
        data=data_yaml,
        epochs=80,
        imgsz=320,  # 640
        batch=4,
        name="run",
        project="yolo11n_0321/",
        device=0
    )

if __name__ == "__main__":
    main()
