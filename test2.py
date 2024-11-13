import os
import cv2
from ultralytics import YOLO

# Initialize EasyOCR reader and YOLO models
helmet_model = YOLO("helmet_detector2.pt")  # Model for detecting helmets

# Output folder for saved images
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Load the target image
image_path = "test5.jpg"  # Replace with the path to your image
frame = cv2.imread(image_path)

# Check if the image was loaded successfully
if frame is None:
    print(f"Error: Image at {image_path} could not be loaded.")
else:
    # Resize the image to fit the screen
    max_width = 1080
    max_height = 680
    height, width = frame.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Detect people in the image using the person model
    detection_results = helmet_model(frame)

    for result in detection_results:
        for i in range(len(result.boxes)):
            box = result.boxes.xyxy[i]
            cls = int(result.boxes.cls[i])
            confidence = result.boxes.conf[i]

            if confidence > 0.5:  # Class 2: Motorbike Rider
                x1, y1, x2, y2 = map(int, box)
                 
                if cls == 2:
                    # Draw a bounding box around the person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (105, 105, 105), 1)
                    cv2.putText(frame, "Rider", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (105, 105, 105), 2)

                if cls == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, "License_plate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if cls == 1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 0), 1)
                    cv2.putText(frame, "Helmet", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 2)                    
                 
    # Show the frame with all drawn boxes for debugging
    cv2.imshow("Annotated Image", frame)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()