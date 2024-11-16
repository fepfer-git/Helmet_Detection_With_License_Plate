import cv2
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
from datetime import datetime
import easyocr
import uuid

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])


def license_complies_format(text):
    return True


def format_license(text):
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
rider_model = YOLO("helmet_detector2.pt")
helmet_model = YOLO("helmet_detector.pt")

# Output folder for saved images
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Load the target image
image_path = "test2.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Image at {image_path} could not be loaded.")
else:
    max_width = 1080
    max_height = 680
    height, width = frame.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    detection_results = rider_model(frame)
    rider_boxes = []

    for result in detection_results:
        for i in range(len(result.boxes)):
            box = result.boxes.xyxy[i]
            cls = int(result.boxes.cls[i])
            confidence = result.boxes.conf[i]

            x1, y1, x2, y2 = map(int, box)
            label = f"{rider_model.names[cls]} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if confidence > 0.1 and cls == 2:
                rider_boxes.append((x1, y1, x2, y2))

    print(f"Number of riders detected: {len(rider_boxes)}")
    print(f"Rider boxes: {rider_boxes}")

    rider_info = {}

    for idx, rider_box in enumerate(rider_boxes):
        print(f"Processing Rider --------- {idx}...")

        x1, y1, x2, y2 = rider_box
        rider_crop = frame[y1:y2, x1:x2]

        detection_results = rider_model(rider_crop)
        helmet_detection_results = helmet_model(rider_crop)

        helmet_found = False
        license_plate = None

        for result in detection_results:
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i]
                cls = int(result.boxes.cls[i])
                confidence = result.boxes.conf[i]

                if confidence > 0.1:
                    bx1, by1, bx2, by2 = map(int, box)
                    bx1 += x1
                    by1 += y1
                    bx2 += x1
                    by2 += y1

                    if cls == 1:
                        helmet_found = True
                    elif cls == 0:
                        license_plate = (bx1, by1, bx2, by2)
                    else:
                        license_plate = (bx1, by1, bx2, by2)

        # Check for helmets with another model
        for result in helmet_detection_results:
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i]
                cls = int(result.boxes.cls[i])
                confidence = result.boxes.conf[i]

                if confidence > 0.1 and cls == 0:
                    helmet_found = True

        if license_plate is not None:
            bx1, by1, bx2, by2 = license_plate
            license_plate_crop = frame[by1:by2, bx1:bx2]

            # Step 1: Convert to grayscale
            gray_plate = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

            # Step 2: Apply Gaussian Blur to reduce noise
            blurred_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)

            # Step 3: Apply Canny edge detection
            edges = cv2.Canny(blurred_plate, 50, 150)

            # Step 4: (Optional) Find contours for further processing
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the original license plate image
            license_plate_contours = license_plate_crop.copy()
            cv2.drawContours(license_plate_contours, contours, -1, (0, 255, 0), 1)

            # Use EasyOCR to read the license plate text
            formatted_license, score = read_license_plate(license_plate_crop)

            rider_info[idx] = {
                "helmet_found": helmet_found,
                "license_plate": formatted_license,  # Save recognized license plate text
                "rider_box": rider_box
            }

            if not helmet_found:
                image_filename = os.path.join(output_folder, f"Rider_{idx}.jpg")
                print(f"Rider {idx} - Helmet Found: {helmet_found}")
                print(f"Detected License Plate: {formatted_license} with confidence: {score:.2f}")

                # Save the image with the bounding boxes
                cv2.imwrite(image_filename, rider_crop)

                # Display the image with contours and recognized text
                cv2.imshow(f"License Plate of Rider {idx} without Helmet", license_plate_contours)
                cv2.waitKey(0)  # Wait until a key is pressed
                cv2.destroyAllWindows()