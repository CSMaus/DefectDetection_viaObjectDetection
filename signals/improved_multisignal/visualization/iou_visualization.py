#!/usr/bin/env python3
"""
IoU (Intersection over Union) Visualization with Real Object Detection
Uses YOLO detection as ground truth and shows how IoU changes with different predictions
"""

from manim import *
import numpy as np
import json
import cv2
from PIL import Image
import os

class IoUVisualization(Scene):
    def construct(self):
        # Title
        title = Text("IoU (Intersection over Union)", font_size=28)
        title.to_edge(UP)
        self.add(title)
        
        # Load detection results
        detection_file = self.find_detection_file()
        if not detection_file:
            error_text = Text("No detection results found!\nRun yolo_detector.py first", 
                            font_size=20, color=RED)
            error_text.move_to(ORIGIN)
            self.add(error_text)
            return
        
        # Load detection data
        with open(detection_file, 'r') as f:
            data = json.load(f)
        
        if not data['detections']:
            error_text = Text("No objects detected in image!", font_size=20, color=RED)
            error_text.move_to(ORIGIN)
            self.add(error_text)
            return
        
        # Load and display the original image
        image_path = data['image_path']
        if os.path.exists(image_path):
            # Load image and create ImageMobject
            pil_image = Image.open(image_path)
            # Resize image to fit in scene
            max_width, max_height = 6, 4
            pil_image.thumbnail((int(max_width * 100), int(max_height * 100)), Image.Resampling.LANCZOS)
            
            # Convert to numpy array for Manim
            img_array = np.array(pil_image)
            image_mob = ImageMobject(img_array)
            image_mob.scale_to_fit_width(max_width)
            image_mob.move_to([0, 0.5, 0])
            
            self.add(image_mob)
        
        # Use the first detection as ground truth
        gt_detection = data['detections'][0]
        image_shape = data['image_shape']
        
        # Scale factor to match image display
        if os.path.exists(image_path):
            display_width = image_mob.width
            display_height = image_mob.height
            scale_x = display_width / image_shape[1]
            scale_y = display_height / image_shape[0]
        else:
            scale_x = scale_y = 6 / max(image_shape[1], image_shape[0])
        
        # Ground truth bbox (scaled and positioned)
        gt_bbox = gt_detection['bbox']
        gt_width = (gt_bbox[2] - gt_bbox[0]) * scale_x
        gt_height = (gt_bbox[3] - gt_bbox[1]) * scale_y
        gt_center_x = ((gt_bbox[0] + gt_bbox[2]) / 2) * scale_x - display_width/2
        gt_center_y = -((gt_bbox[1] + gt_bbox[3]) / 2) * scale_y + display_height/2
        
        gt_rect = Rectangle(
            width=gt_width,
            height=gt_height,
            color=GREEN,
            fill_opacity=0.3,
            stroke_width=3
        )
        gt_rect.move_to([gt_center_x, gt_center_y + 0.5, 0])
        
        # Labels with backgrounds for visibility
        gt_label = Text(f"Ground Truth: {gt_detection['class_name']}", 
                       font_size=16, color=GREEN)
        gt_label.move_to([0, 2.5, 0])
        gt_bg = Rectangle(width=gt_label.width + 0.2, height=gt_label.height + 0.1, 
                         color=BLACK, fill_opacity=0.8)
        gt_bg.move_to(gt_label.get_center())
        
        pred_label = Text("Predicted", font_size=16, color=RED)
        pred_label.move_to([0, 2, 0])
        pred_bg = Rectangle(width=pred_label.width + 0.2, height=pred_label.height + 0.1, 
                           color=BLACK, fill_opacity=0.8)
        pred_bg.move_to(pred_label.get_center())
        
        iou_text = Text("IoU = 0.00", font_size=20, color=YELLOW)
        iou_text.move_to([0, -2.5, 0])
        iou_bg = Rectangle(width=iou_text.width + 0.2, height=iou_text.height + 0.1, 
                          color=BLACK, fill_opacity=0.8)
        iou_bg.move_to(iou_text.get_center())
        
        # Formula positioned below IoU value with background
        formula = Text("IoU = Area of Intersection / Area of Union", font_size=14)
        formula.move_to([0, -3, 0])
        formula_bg = Rectangle(width=formula.width + 0.2, height=formula.height + 0.1, 
                              color=BLACK, fill_opacity=0.8)
        formula_bg.move_to(formula.get_center())
        
        # Show ground truth
        self.play(Create(gt_rect), Create(gt_bg), Write(gt_label))
        self.wait(1)
        
        # Create predicted bbox (initially no overlap)
        pred_rect = Rectangle(
            width=gt_width * 0.8,
            height=gt_height * 0.8,
            color=RED,
            fill_opacity=0.2,
            stroke_width=3
        )
        pred_rect.move_to(gt_rect.get_center() + np.array([gt_width + 0.5, 0, 0]))
        
        self.play(Create(pred_rect), Create(pred_bg), Write(pred_label), 
                 Create(iou_bg), Write(iou_text))
        self.play(Create(formula_bg), Write(formula))
        self.wait(1)
        
        # Show different IoU scenarios with ACTUAL overlaps
        scenarios = [
            {"name": "No Overlap", "offset": [gt_width + 0.5, 0], "scale": 0.8, "expected_iou": 0.0},
            {"name": "Small Overlap", "offset": [gt_width * 0.4, 0], "scale": 0.8, "expected_iou": 0.2},
            {"name": "Moderate Overlap", "offset": [gt_width * 0.2, gt_height * 0.1], "scale": 0.9, "expected_iou": 0.4},
            {"name": "Good Overlap", "offset": [gt_width * 0.1, gt_height * 0.05], "scale": 0.95, "expected_iou": 0.7},
            {"name": "Excellent Match", "offset": [0, 0], "scale": 1.0, "expected_iou": 0.9}
        ]
        
        for scenario in scenarios:
            # Calculate new position and size
            new_width = gt_width * scenario["scale"]
            new_height = gt_height * scenario["scale"]
            new_center = gt_rect.get_center() + np.array([scenario["offset"][0], scenario["offset"][1], 0])
            
            # Create new predicted rectangle
            new_pred_rect = Rectangle(
                width=new_width,
                height=new_height,
                color=RED,
                fill_opacity=0.2,
                stroke_width=3
            )
            new_pred_rect.move_to(new_center)
            
            # Calculate actual IoU
            iou_value = self.calculate_iou_from_rects(gt_rect, new_pred_rect)
            
            # Update IoU text with background
            new_iou_text = Text(f"IoU = {iou_value:.2f}", font_size=20, color=YELLOW)
            new_iou_text.move_to([0, -2.5, 0])
            new_iou_bg = Rectangle(width=new_iou_text.width + 0.2, height=new_iou_text.height + 0.1, 
                                  color=BLACK, fill_opacity=0.8)
            new_iou_bg.move_to(new_iou_text.get_center())
            
            # Scenario label with background
            scenario_text = Text(scenario["name"], font_size=14, color=WHITE)
            scenario_text.move_to([0, 1.5, 0])
            scenario_bg = Rectangle(width=scenario_text.width + 0.2, height=scenario_text.height + 0.1, 
                                   color=BLACK, fill_opacity=0.8)
            scenario_bg.move_to(scenario_text.get_center())
            
            # Animate transition
            self.play(
                Transform(pred_rect, new_pred_rect),
                Transform(iou_text, new_iou_text),
                Transform(iou_bg, new_iou_bg),
                Transform(pred_label, scenario_text),
                Transform(pred_bg, scenario_bg),
                run_time=2
            )
            self.wait(1.5)
        
        self.wait(3)
    
    def find_detection_file(self):
        """Find the most recent detection JSON file"""
        detection_dir = "detection_results"
        if not os.path.exists(detection_dir):
            return None
        
        json_files = [f for f in os.listdir(detection_dir) if f.endswith('_detections.json')]
        if not json_files:
            return None
        
        # Return the most recent file
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(detection_dir, x)), reverse=True)
        return os.path.join(detection_dir, json_files[0])
    
    def scale_bbox(self, bbox, scale_factor):
        """Scale bbox coordinates"""
        return [coord * scale_factor for coord in bbox]
    
    def calculate_iou_from_rects(self, rect1, rect2):
        """Calculate IoU between two Manim rectangles"""
        # Get rectangle bounds
        r1_left = rect1.get_center()[0] - rect1.width / 2
        r1_right = rect1.get_center()[0] + rect1.width / 2
        r1_top = rect1.get_center()[1] + rect1.height / 2
        r1_bottom = rect1.get_center()[1] - rect1.height / 2
        
        r2_left = rect2.get_center()[0] - rect2.width / 2
        r2_right = rect2.get_center()[0] + rect2.width / 2
        r2_top = rect2.get_center()[1] + rect2.height / 2
        r2_bottom = rect2.get_center()[1] - rect2.height / 2
        
        # Calculate intersection
        inter_left = max(r1_left, r2_left)
        inter_right = min(r1_right, r2_right)
        inter_top = min(r1_top, r2_top)
        inter_bottom = max(r1_bottom, r2_bottom)
        
        if inter_left < inter_right and inter_bottom < inter_top:
            intersection = (inter_right - inter_left) * (inter_top - inter_bottom)
        else:
            intersection = 0
        
        # Calculate union
        area1 = rect1.width * rect1.height
        area2 = rect2.width * rect2.height
        union = area1 + area2 - intersection
        
        # Calculate IoU
        if union == 0:
            return 0
        return intersection / union


class IoUInteractiveVisualization(Scene):
    """Interactive IoU visualization with smooth transitions"""
    
    def construct(self):
        title = Text("Interactive IoU Demonstration", font_size=28)
        title.to_edge(UP)
        self.add(title)
        
        # Create ground truth bbox
        gt_rect = Rectangle(width=3, height=2, color=GREEN, fill_opacity=0.3, stroke_width=3)
        gt_rect.move_to([0, 0.5, 0])
        
        # Create predicted bbox
        pred_rect = Rectangle(width=2, height=1.5, color=RED, fill_opacity=0.2, stroke_width=3)
        pred_rect.move_to([2, 1.5, 0])
        
        # Labels
        gt_label = Text("Ground Truth", font_size=16, color=GREEN)
        gt_label.next_to(gt_rect, DOWN)
        
        pred_label = Text("Prediction", font_size=16, color=RED)
        pred_label.next_to(pred_rect, DOWN)
        
        # IoU display
        iou_text = Text("IoU = 0.00", font_size=24, color=YELLOW)
        iou_text.move_to([0, -2.5, 0])
        
        # Show initial setup
        self.play(Create(gt_rect), Write(gt_label))
        self.play(Create(pred_rect), Write(pred_label))
        self.play(Write(iou_text))
        self.wait(1)
        
        # Smooth animation showing IoU changes
        def update_iou(mob, alpha):
            # Interpolate position and size
            target_pos = gt_rect.get_center()
            current_pos = pred_rect.get_center()
            new_pos = current_pos + alpha * (target_pos - current_pos)
            
            target_width = gt_rect.width
            target_height = gt_rect.height
            new_width = pred_rect.width + alpha * (target_width - pred_rect.width)
            new_height = pred_rect.height + alpha * (target_height - pred_rect.height)
            
            # Update rectangle
            new_rect = Rectangle(width=new_width, height=new_height, 
                               color=RED, fill_opacity=0.2, stroke_width=3)
            new_rect.move_to(new_pos)
            mob.become(new_rect)
            
            # Calculate and update IoU
            iou_value = self.calculate_iou_from_rects(gt_rect, mob)
            new_iou_text = Text(f"IoU = {iou_value:.2f}", font_size=24, color=YELLOW)
            new_iou_text.move_to([0, -2.5, 0])
            iou_text.become(new_iou_text)
        
        # Animate smooth transition
        self.play(UpdateFromAlphaFunc(pred_rect, update_iou), run_time=5, rate_func=smooth)
        self.wait(2)
    
    def calculate_iou_from_rects(self, rect1, rect2):
        """Calculate IoU between two Manim rectangles"""
        # Get rectangle bounds
        r1_left = rect1.get_center()[0] - rect1.width / 2
        r1_right = rect1.get_center()[0] + rect1.width / 2
        r1_top = rect1.get_center()[1] + rect1.height / 2
        r1_bottom = rect1.get_center()[1] - rect1.height / 2
        
        r2_left = rect2.get_center()[0] - rect2.width / 2
        r2_right = rect2.get_center()[0] + rect2.width / 2
        r2_top = rect2.get_center()[1] + rect2.height / 2
        r2_bottom = rect2.get_center()[1] - rect2.height / 2
        
        # Calculate intersection
        inter_left = max(r1_left, r2_left)
        inter_right = min(r1_right, r2_right)
        inter_top = min(r1_top, r2_top)
        inter_bottom = max(r1_bottom, r2_bottom)
        
        if inter_left < inter_right and inter_bottom < inter_top:
            intersection = (inter_right - inter_left) * (inter_top - inter_bottom)
        else:
            intersection = 0
        
        # Calculate union
        area1 = rect1.width * rect1.height
        area2 = rect2.width * rect2.height
        union = area1 + area2 - intersection
        
        # Calculate IoU
        if union == 0:
            return 0
        return intersection / union
