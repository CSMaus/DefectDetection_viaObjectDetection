from ultralytics import YOLO

def main():
    model_path = "yolo9c-seg/segmentation320/weights/best.pt"
    model = YOLO(model_path)

    # Run segmentation on an image (save=True saves an annotated result image)
    results = model.predict("787-225_01_Ch-0_51.png", save=True)
    print(results)

if __name__ == "__main__":
    main()
