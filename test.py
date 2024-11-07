import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("license_plate_detector.pt")

# Run inference on the video file
results = model(source="4.mp4", show=True, conf=0.4, save=True)

# Iterate over the detected objects
for result in results:
    # Get the class labels, confidence scores, and bounding boxes
    boxes = result.boxes.xyxy  # Bounding boxes in (x1, y1, x2, y2) format
    confidences = result.boxes.conf  # Confidence scores
    class_ids = result.boxes.cls  # Class IDs

    # Print the information
    for i in range(len(boxes)):
        class_id = int(class_ids[i])
        class_name = model.names[class_id]
        print(f"Class ID: {class_id}, Class Name: {class_name}, Confidence: {confidences[i]}, Box: {boxes[i]}")