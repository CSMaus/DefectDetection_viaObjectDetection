# here is the code to test the correctness of annotations for created b-scan images
import sys
import os
import json
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QSlider
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from PyQt6.QtCore import Qt

COLLECTION_NAME = "WOT-20250522(auto)"
DATASET_DIR = "dataset/predictions"
# f"dataset/{COLLECTION_NAME}"
ANNOTATIONS_FILE = f"annotations-{COLLECTION_NAME}.json"
IMAGE_SIZE = (320, 320)


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BBox Viewer")

        self.folder_label = QLabel("Folder:")
        self.image_label = QLabel("Image:")
        self.image_display = QLabel()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.update_image)

        layout = QVBoxLayout()
        layout.addWidget(self.folder_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.image_display)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.images = []  # (folder, image_name)
        self.annotations = self.load_annotations()
        self.load_all_images()

        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.images) - 1)

        if self.images:
            self.update_image(0)

    def load_annotations(self):
        with open(ANNOTATIONS_FILE, 'r') as f:
            return json.load(f)

    def load_all_images(self):
        for folder in os.listdir(DATASET_DIR):
            folder_path = os.path.join(DATASET_DIR, folder)
            if not os.path.isdir(folder_path):
                continue
            for img_file in sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0])):
                if img_file.endswith(".png"):
                    self.images.append((folder, img_file))
        print("Total number of images is: ", len(self.images))

    def update_image(self, idx):
        folder, img_name = self.images[idx]
        self.folder_label.setText(f"Folder: {folder}")
        self.image_label.setText(f"Image: {img_name}")

        img_path = os.path.join(DATASET_DIR, folder, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        painter = QPainter(pixmap)

        defects = self.annotations.get(folder, {}).get(img_name, [])
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        painter.setFont(QFont("Arial", 10))

        for defect in defects:
            x1, x2, y1, y2 = defect["bbox"]
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            painter.drawRect(x, y, w, h)
            painter.drawText(x + 2, y - 4, defect["label"])

        painter.end()
        self.image_display.setPixmap(pixmap.scaled(*IMAGE_SIZE, Qt.AspectRatioMode.KeepAspectRatio))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_P:
            idx = self.slider.value()
            folder, img_name = self.images[idx]
            print(f"{folder}, scan: {img_name}")

            save_dir = os.path.join("bad_samples", folder)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, img_name)

            img_path = os.path.join(DATASET_DIR, folder, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            painter = QPainter(pixmap)
            defects = self.annotations.get(folder, {}).get(img_name, [])
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.setFont(QFont("Arial", 10))

            for defect in defects:
                x1, x2, y1, y2 = defect["bbox"]
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                painter.drawRect(x, y, w, h)
                painter.drawText(x + 2, y - 4, defect["label"])
            painter.end()
            pixmap.save(save_path)

        elif event.key() == Qt.Key.Key_Right:
            idx = self.slider.value()
            if idx < len(self.images) - 1:
                self.slider.setValue(idx + 1)

        elif event.key() == Qt.Key.Key_Left:
            idx = self.slider.value()
            if idx > 0:
                self.slider.setValue(idx - 1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.resize(IMAGE_SIZE[0], IMAGE_SIZE[1] + 80)
    viewer.show()
    sys.exit(app.exec())
