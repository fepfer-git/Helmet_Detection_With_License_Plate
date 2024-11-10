import os
import cv2
from ultralytics import YOLO
import easyocr

# Initialize EasyOCR reader and YOLO models
reader = easyocr.Reader(['en'])
helmet_model = YOLO("helmet_detector.pt")  # Model for detecting helmets
license_plate_model = YOLO("license_plate_detector.pt")  # Model for detecting license plates
person_model = YOLO("yolov8n.pt")  # Model to detect people (for getting full person and bike)

# Output folder for saved images
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

def extract_license_text(crop_img):
    """
    Extracts text from the cropped license plate image using EasyOCR.
    """
    detections = reader.readtext(crop_img)
    if not detections:
        print("No text detected in the license plate region.")
        return None

    for bbox, text, score in detections:
        text = text.upper().replace(" ", "")
        print(f"Detected text: {text} with score {score}")
        if len(text) >= 5:  # Example condition for a valid license plate format
            return text
    return None

# Load the target image
image_path = "test5.jpg"  # Replace with the path to your image
frame = cv2.imread(image_path)

# Check if the image was loaded successfully
if frame is None:
    print(f"Error: Image at {image_path} could not be loaded.")
else:
    # Detect people in the image using the person model
    person_results = person_model(frame)
    save_with_license = False  # Flag to check if a no-helmet person is found
    license_text_for_save = "having_helmet"  # Default name if everyone has helmets

    for result in person_results:
        for i in range(len(result.boxes)):
            box = result.boxes.xyxy[i]
            cls = int(result.boxes.cls[i])
            confidence = result.boxes.conf[i]

            if cls == 0 and confidence > 0.5:  # Class 0: Person
                x1, y1, x2, y2 = map(int, box)
                
                # Expand the box horizontally to capture the bike and rider
                padding = 30  # Increase the box size by 10 pixels on each side
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                # Crop the region with the expanded area
                person_roi = frame[y1:y2, x1:x2]
                print(f"Detected person with expanded region: {x1, y1, x2, y2}")

                # Draw a bounding box around the person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (105, 105, 105), 1)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (105, 105, 105), 2)

                # Run helmet detection within the expanded region
                helmet_results = helmet_model(person_roi)
                
                for helmet_result in helmet_results:
                    for j in range(len(helmet_result.boxes)):
                        helmet_box = helmet_result.boxes.xyxy[j]
                        helmet_cls = int(helmet_result.boxes.cls[j])
                        helmet_conf = helmet_result.boxes.conf[j]

                        hx1, hy1, hx2, hy2 = map(int, helmet_box)
                        label = "No-Helmet" if helmet_cls == 1 else "Helmet"

                        # Run license plate detection within the expanded region if no helmet
                        if helmet_cls == 0:  # No helmet detected
                            cv2.rectangle(frame, (x1 + hx1, y1 + hy1), (x1 + hx2, y1 + hy2), (0, 255, 0), 2)  # Green color in BGR
                            cv2.putText(frame, label, (x1 + hx1, y1 + hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                                     
                            save_with_license = True  # We should save based on license plate
                            license_plate_results = license_plate_model(person_roi)

                            for lp_result in license_plate_results:
                                if len(lp_result.boxes) > 0:
                                    # Use the first detected license plate in the person's ROI
                                    lp_box = lp_result.boxes.xyxy[0]
                                    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box)

                                    # Draw bounding box for the detected license plate
                                    cv2.rectangle(frame, (x1 + lp_x1, y1 + lp_y1), (x1 + lp_x2, y1 + lp_y2), (220,20,60), 2)
                                    cv2.putText(frame, "License Plate", (x1 + lp_x1, y1 + lp_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,20,60), 2)

                                    # Extract license plate region and read text
                                    license_plate_roi = person_roi[lp_y1:lp_y2, lp_x1:lp_x2]
                                    license_text = extract_license_text(license_plate_roi)

                                    if license_text:
                                        license_text_for_save = license_text  # Update to use license plate text

                        if helmet_cls == 1: 
                            cv2.rectangle(frame, (x1 + hx1, y1 + hy1), (x1 + hx2, y1 + hy2), (255, 0, 0), 2)
                            cv2.putText(frame, label, (x1 + hx1, y1 + hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame with all drawn boxes for debugging
    cv2.imshow("Annotated Image", frame)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

    # Save the image with the appropriate filename
    image_filename = os.path.join(output_folder, f"{license_text_for_save}.jpg")
    cv2.imwrite(image_filename, frame)
    print(f"Saved image to {image_filename}")
