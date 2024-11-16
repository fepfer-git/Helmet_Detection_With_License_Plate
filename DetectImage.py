import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
from datetime import datetime
import easyocr
import uuid

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # You can specify multiple languages if needed

def license_complies_format(text):
    # Implement your logic to check if the license plate text complies with the desired format
    return True

def format_license(text):
    # Implement your logic to format the license plate text
    return text

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None

# Load the YOLO models
rider_model = YOLO("helmet_detector2.pt")  # Model for detecting helmets
helmet_model = YOLO("helmet_detector.pt")  # Model for detecting helmets

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
    # Detect riders in the image
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

            # Draw bounding box and label for all detected objects
            x1, y1, x2, y2 = map(int, box)
            label = f"{rider_model.names[cls]} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if confidence > 0.1 and cls == 2:  # Class 2: Motorbike Rider
                rider_boxes.append((x1, y1, x2, y2))

    print(f"Number of riders detected: {len(rider_boxes)}")
    print(f"Rider boxes: {rider_boxes}")

    # Dictionary to store rider boxes and their corresponding helmet and license plate
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

                    if cls == 1:  # Class 1: Helmet
                        helmet_found = True
                    elif cls == 0:  # Class 0: License Plate
                        license_plate = (bx1, by1, bx2, by2)
                        
        # Check again with another model
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

            rider_info[idx] = {
                "helmet_found": helmet_found,
                "license_plate": license_plate,
                "rider_box": rider_box
            }
            
            if not helmet_found:
                image_filename = os.path.join(output_folder, f"Rider_{idx}.jpg")
                print(f"Rider {idx} - Helmet Found: {helmet_found}")
                
                # Save the image with the bounding boxes
                cv2.imwrite(image_filename, rider_crop)
                
                # Display the image with the bounding boxes
                cv2.imshow(f"License Plate of Rider {idx} without Helmet", rider_crop)
                cv2.waitKey(0)  # Wait until a key is pressed
                cv2.destroyAllWindows()