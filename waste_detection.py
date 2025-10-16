import torch
import cv2
import numpy as np
from PIL import Image

class WasteDetector:
    def __init__(self, weights_path='best.pt', img_size=640):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        self.model.conf = 0.5  # Confidence threshold
        self.img_size = img_size
        
        # Define waste categories and their colors for visualization
        self.categories = {
            0: ('paper', (255, 255, 0)),        # Yellow
            1: ('plastic', (255, 0, 0)),        # Blue
            2: ('glass', (0, 255, 0)),          # Green
            3: ('metal', (128, 128, 128)),      # Gray
            4: ('plastic_bottle', (255, 0, 255)),# Magenta
            5: ('glass_bottle', (0, 255, 255)),  # Cyan
            6: ('biodegradable', (0, 128, 0)),   # Dark Green
            7: ('cardboard', (165, 42, 42))      # Brown
        }

    def detect_waste(self, frame):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make detection
        results = self.model(frame_rgb, size=self.img_size)
        
        # Process results
        detections = results.xyxy[0].cpu().numpy()
        
        # Draw bounding boxes and labels
        annotated_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            
            if conf >= self.model.conf:
                # Get category name and color
                category, color = self.categories[int(class_id)]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            color, 2)
                
                # Create label with category and confidence
                label = f'{category} {conf:.2f}'
                
                # Calculate text size and position
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(annotated_frame,
                            (int(x1), int(y1) - label_height - baseline - 10),
                            (int(x1) + label_width, int(y1)),
                            color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label,
                           (int(x1), int(y1) - baseline - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Initialize detector
    detector = WasteDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection and get annotated frame
        annotated_frame = detector.detect_waste(frame)
        
        # Display result
        cv2.imshow('Waste Segregation Detection', annotated_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()