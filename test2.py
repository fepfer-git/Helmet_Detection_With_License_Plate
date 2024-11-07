import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
from datetime import datetime
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # You can specify multiple languages if needed

def license_complies_format(text):
    # Implement your logic to check if the license plate text complies with the desired format
    return True

def format_license(text):
    # Implement your logic to format the license plate text
    return text

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score

    return None, None

# Load the YOLO models
helmet_model = YOLO("helmet_detector.pt")
license_plate_model = YOLO("license_plate_detector.pt")
yolov8n = YOLO("yolov8n.pt")

# Use GPU
helmet_model.to('cuda:0')
license_plate_model.to('cuda:0')
yolov8n.to('cuda:0')

# Ensure the violation folder exists
violation_folder = "violation"
os.makedirs(violation_folder, exist_ok=True)

# Set to keep track of saved license plates
saved_license_plates = set()

# Open the video file
video_path = "HANO.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame for helmet detection
        yolov8n_results = yolov8n(frame)
        
        for yolov8n_result in yolov8n_results:
            # Get the class labels, confidence scores, and bounding boxes
            yolov8n_boxes = yolov8n_result.boxes.xyxy
            yolov8n_confidences = yolov8n_result.boxes.conf
            yolov8n_class_ids = yolov8n_result.boxes.cls
            
            for i in range(len(yolov8n_boxes)):
                yolov8n_class_id = int(yolov8n_class_ids[i])
                
                # Filter to only track person (class_id 0) and motorcycle (class_id 3)
                if yolov8n_class_id in [0]:
                    yolov8n_class_name = yolov8n_result.names[yolov8n_class_id]
                    print(f"Class: {yolov8n_class_name}, Confidence: {yolov8n_confidences[i]}, Box: {yolov8n_boxes[i]}")
                    x1, y1, x2, y2 = map(int, yolov8n_boxes[i])
                    
                    # Increase the size of the bounding box for person (class_id 0)
                    if yolov8n_class_id == 0:
                        padding = 30  # Increase the box size by 10 pixels on each side
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(frame.shape[1], x2 + padding)
                        y2 = min(frame.shape[0], y2 + padding)
                    
                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (192,192,192), 1)
                    cv2.putText(frame, yolov8n_class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (192,192,192), 2)          
                    
                    if yolov8n_class_id == 0:
                        roi = frame[y1:y2, x1:x2]
                        
                        helmet_results = helmet_model(roi)
                        for helmet_result in helmet_results:
                            helmet_boxes = helmet_result.boxes.xyxy
                            helmet_class_ids = helmet_result.boxes.cls
                            helmet_confidences = helmet_result.boxes.conf
                            
                            for j in range(len(helmet_boxes)):
                                helmet_class_id = int(helmet_class_ids[j])
                                helmet_class_name = helmet_model.names[helmet_class_id]
                                confidence = helmet_confidences[j]
                                print(f"Class: {helmet_class_name}, Confidence: {confidence}, Box: {helmet_boxes[j]}")
                                
                                x1_h, y1_h, x2_h, y2_h = map(int, helmet_boxes[j])
                                label = f"{helmet_class_name} {confidence:.2f}"
                                
                                if confidence > 0.5:
                                    if helmet_class_id == 1:
                                        cv2.rectangle(frame, (x1 + x1_h, y1 + y1_h), (x1 + x2_h, y1 + y2_h), (255, 0, 0), 2)
                                        cv2.putText(frame, label, (x1 + x1_h, y1 + y1_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                                        license_plate_results = license_plate_model(roi)
                                        if len(license_plate_results) > 0:
                                            license_plate_result = license_plate_results[0]
                                            license_plate_boxes = license_plate_result.boxes.xyxy
                                            
                                            if len(license_plate_boxes) > 0:
                                                lp_x1, lp_y1, lp_x2, lp_y2 = map(int, license_plate_boxes[0])
                                                cv2.rectangle(frame, (x1 + lp_x1, y1 + lp_y1), (x1 + lp_x2, y1 + lp_y2), (0, 0, 255), 2)
                                                cv2.putText(frame, "License Plate", (x1 + lp_x1, y1 + lp_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                                
                                                # Extract the ROI for the license plate
                                                license_plate_roi = frame[y1 + lp_y1:y1 + lp_y2, x1 + lp_x1:x1 + lp_x2]
                                                
                                                # Use the read_license_plate function to read the text from the license plate ROI
                                                license_plate_text, score = read_license_plate(license_plate_roi)
                                                print(f"License Plate Text: {license_plate_text}, Score: {score}")
                                                
                                                # Save the frame to the violation folder with the license plate text as the filename
                                                if license_plate_text and license_plate_text not in saved_license_plates:
                                                    saved_license_plates.add(license_plate_text)
                                                    frame_filename = os.path.join(violation_folder, f"{license_plate_text}.jpg")
                                                    cv2.imwrite(frame_filename, frame)
                                                    print(f"Saved frame to {frame_filename}")                                   
                                                
                                    if helmet_class_id == 0:
                                        cv2.rectangle(frame, (x1 + x1_h, y1 + y1_h), (x1 + x2_h, y1 + y2_h), (51, 204, 51), 2)
                                        cv2.putText(frame, label, (x1 + x1_h, y1 + y1_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (51, 204, 51), 2)
                                
        # Display the annotated frame
        cv2.imshow("YOLO Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()