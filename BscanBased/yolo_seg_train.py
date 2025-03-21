from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    data_yaml = "data-seg.yaml"  # segmentation dataset setup .yaml
    model = YOLO("yolov9c-seg.yaml")
    model = YOLO("yolov9c-seg.pt")

    # 'project' sets the folder where results go
    # 'name' is the subfolder name; final weights are in project/name/weights/best.pt
    model.train(
        data=data_yaml,
        epochs=30,
        imgsz=320,
        project="yolo9c-seg/",
        name="segmentation",
        device=0
    )

if __name__ == "__main__":
    main()
