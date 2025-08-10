#!/usr/bin/env python3
"""
YOLO Object Detection Script
Detects objects in real images and saves detection results
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

class YOLODetector:
    def __init__(self):
        """Initialize YOLO detector"""
        # Try to use YOLOv5 from ultralytics (pip install ultralytics)
        try:
            import torch
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')  # nano model for speed
            self.available = True
            print("✅ YOLO model loaded successfully")
        except ImportError:
            print("❌ ultralytics not installed. Install with: pip install ultralytics")
            self.available = False
        except Exception as e:
            print(f"❌ Error loading YOLO: {e}")
            self.available = False
    
    def detect_objects(self, image_path, confidence_threshold=0.5):
        """
        Detect objects in image
        Returns: list of detections with bbox coordinates and class info
        """
        if not self.available:
            return None
        
        try:
            # Run inference
            results = self.model(image_path, conf=confidence_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bbox coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name,
                            'width': float(x2 - x1),
                            'height': float(y2 - y1),
                            'center_x': float((x1 + x2) / 2),
                            'center_y': float((y1 + y2) / 2)
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return None
    
    def save_detection_results(self, image_path, output_dir="detection_results"):
        """
        Detect objects and save results (image + JSON)
        """
        if not self.available:
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Get detections
        detections = self.detect_objects(image_path)
        if not detections:
            print("❌ No objects detected")
            return None
        
        # Draw bboxes on image
        annotated_image = image.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            
            # Draw bbox
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save annotated image
        image_name = Path(image_path).stem
        output_image_path = os.path.join(output_dir, f"{image_name}_detected.jpg")
        cv2.imwrite(output_image_path, annotated_image)
        
        # Save detection data
        detection_data = {
            'image_path': image_path,
            'image_shape': image.shape,
            'detections': detections
        }
        
        output_json_path = os.path.join(output_dir, f"{image_name}_detections.json")
        with open(output_json_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        print(f"✅ Saved detection results:")
        print(f"   Image: {output_image_path}")
        print(f"   Data: {output_json_path}")
        print(f"   Found {len(detections)} objects")
        
        return {
            'image_path': output_image_path,
            'json_path': output_json_path,
            'detections': detections,
            'image_shape': image.shape
        }

def main():
    """Test the detector with the dog_bike_car image"""
    detector = YOLODetector()
    
    if not detector.available:
        print("YOLO not available. Please install ultralytics:")
        print("pip install ultralytics")
        return
    
    # Look for dog_bike_car image with common extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    test_image = None
    
    for ext in image_extensions:
        image_path = f"dog_bike_car{ext}"
        if os.path.exists(image_path):
            test_image = image_path
            break
    
    if not test_image:
        print("❌ dog_bike_car image not found!")
        print("   Looking for: dog_bike_car.jpg, dog_bike_car.png, etc.")
        print("   Make sure the image is in the current directory")
        return
    
    print(f"✅ Found image: {test_image}")
    
    # Run detection
    results = detector.save_detection_results(test_image)
    if results:
        print(f"\n🎯 Detection completed! Found objects:")
        for i, det in enumerate(results['detections']):
            print(f"  {i+1}. {det['class_name']} (confidence: {det['confidence']:.2f})")
        print(f"\n📁 Results saved in 'detection_results/' folder")
        print(f"🎬 Now you can run IoU visualizations:")
        print(f"   python run_visualizations.py 3")
        print(f"   python run_visualizations.py 4")

if __name__ == "__main__":
    main()
