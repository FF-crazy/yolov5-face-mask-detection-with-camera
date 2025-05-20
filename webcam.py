import cv2
import torch
import numpy as np
import time
from pathlib import Path

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Load model
model_path = Path('models/mask_yolov5.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.to(device)
model.eval()

# Class names
class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Green for with mask, Red for without mask, Yellow for incorrect

def process_frame(frame):
    # Inference
    results = model(frame)
    
    # Process results
    for *xyxy, conf, cls in results.xyxy[0]:  # xyxy, confidence, class
        if conf > 0.5:  # confidence threshold
            # Get coordinates and class
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            class_idx = int(cls)
            label = f'{class_names[class_idx]} {conf:.2f}'
            
            # Draw bounding box
            color = colors[class_idx]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam opened successfully. Press 'q' to quit.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Process frame
        start_time = time.time()
        processed_frame = process_frame(frame)
        end_time = time.time()
        
        # Calculate FPS
        fps = 1 / (end_time - start_time)
        cv2.putText(processed_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Mask Detection', processed_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()