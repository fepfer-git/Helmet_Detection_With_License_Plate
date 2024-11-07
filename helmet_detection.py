import cv2
import os
from ultralytics import YOLO
from datetime import datetime
import pytesseract

# Configure pytesseract to use the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary

# Load the YOLO models
helmet_model = YOLO("helmet_detector.pt")
license_plate_model = YOLO("license_plate_detector.pt")

# Print class names and their corresponding IDs for the helmet model
helmet_class_names = helmet_model.names
for class_id, class_name in helmet_class_names.items():
    print(f"Helmet Class ID: {class_id}, Class Name: {class_name}")

# Ensure the violation folder exists
violation_folder = "violation"
os.makedirs(violation_folder, exist_ok=True)

# Open the video file
video_path = "TEST_LICENSE.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame for helmet detection
        helmet_results = helmet_model(frame)
        license_plate_results = license_plate_model(frame)
        
        # Iterate over the detected objects for helmet detection
        for helmet_result in helmet_results:
            # Get the class labels, confidence scores, and bounding boxes
            helmet_boxes = helmet_result.boxes.xyxy  # Bounding boxes in (x1, y1, x2, y2) format
            helmet_confidences = helmet_result.boxes.conf  # Confidence scores
            helmet_class_ids = helmet_result.boxes.cls  # Class IDs

            # Print the information
            for i in range(len(helmet_boxes)):
                helmet_class_id = int(helmet_class_ids[i])
                helmet_class_name = helmet_class_names[helmet_class_id]
                print(f"Class: {helmet_class_name}, Confidence: {helmet_confidences[i]}, Box: {helmet_boxes[i]}")
                if helmet_class_id == 1:
                    x1, y1, x2, y2 = map(int, helmet_boxes[i])  # Convert coordinates to integers
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, helmet_class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    

        # Visualize the results on the frame
        annotated_frame = helmet_results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()