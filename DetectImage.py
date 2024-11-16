import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from matplotlib import pyplot as plt
from ultralytics import YOLO
import easyocr

def read_license_plate(license_plate_crop):
    print("Reading License Plate") 
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.show()

    # Use EasyOCR to read text
    reader = easyocr.Reader(['en'])
    results = reader.readtext(license_plate_crop)

    license_plate_text = ''
    print(results)
    if results:
        for i in results :
            license_plate_text = license_plate_text + ' ' + i[1]
    
    print(license_plate_text)
    return license_plate_text

# Load the YOLO models
rider_model = YOLO("helmet_detector2.pt")  # Model for detecting helmets
helmet_model = YOLO("helmet_detector.pt")  # Model for detecting helmets

# Output folder for saved images
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Load the target image
image_path = "test_img.jpg"  # Replace with the path to your image
frame = cv2.imread(image_path)

# Check if the image was loaded successfully
if frame is None:
    print(f"Error: Image at {image_path} could not be loaded.")
else:
    # Keep a copy of the original frame
    showed_frame = frame.copy()
    print("Detecting Riders")
    detection_results = rider_model(frame)
    rider_boxes = []
        
    for result in detection_results:
        for i in range(len(result.boxes)):
            box = result.boxes.xyxy[i]
            cls = int(result.boxes.cls[i])
            confidence = result.boxes.conf[i]
            padding = 15
            x1, y1, x2, y2 = map(int, box) #Saved data
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            showx1 = max(0, x1 - padding)
            showy1 = max(0, y1 - padding)
            showx2 = min(frame.shape[1], x2 + padding)
            showy2 = min(frame.shape[0], y2 + padding)
                       
            showx1, showy1, showx2, showy2 = map(int, box) #Show data
            if cls == 2:
                cv2.rectangle(showed_frame, (showx1, showy1), (showx2, showy2), (0, 255, 0), 2)
                cv2.putText(showed_frame, "Rider", (showx1, showy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if cls == 1:
                cv2.rectangle(showed_frame, (showx1, showy1), (showx2, showy2), (0, 0, 255), 2)
                cv2.putText(showed_frame, "Helmet", (showx1, showy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if cls == 0:
                cv2.rectangle(showed_frame, (showx1, showy1), (showx2, showy2), (255, 0, 0), 2)
                cv2.putText(showed_frame, "License Plate", (showx1, showy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            full_frame_filename = os.path.join(output_folder, f"Detected_Full.jpg")
            cv2.imwrite(full_frame_filename, showed_frame)

            if confidence > 0.1 and cls == 2:  # Class 2: Motorbike Rider
                rider_boxes.append((x1, y1, x2, y2))

    # Dictionary to store rider boxes and their corresponding helmet and license plate
    rider_info = {}

    for idx, rider_box in enumerate(rider_boxes):        
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
                        
        # for result in helmet_detection_results:
        #     for i in range(len(result.boxes)):
        #         box = result.boxes.xyxy[i]
        #         cls = int(result.boxes.cls[i])
        #         confidence = result.boxes.conf[i]

        #         if confidence > 0.1 and cls == 0:
        #             helmet_found = True
        
        print(f"Rider {idx}: Helmet Found: {helmet_found}, License Plate: {license_plate}")
        license_text = 'Not Found'
        if (license_plate is not None):
            bx1, by1, bx2, by2 = license_plate
            # Crop the license plate region from the original frame
            license_plate_crop = frame[by1:by2, bx1:bx2]
            license_text = read_license_plate(license_plate_crop)
        
        if not helmet_found:
            image_filename = os.path.join(output_folder, f"Rider_{idx}_License_Plate_{license_text}.jpg")
            
            # Draw the bounding box around the license plate
            cv2.rectangle(rider_crop, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), (138,43,226), 1)
            cv2.putText(rider_crop, f"{license_text}", (bx1 - x1, by1 - y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (138,43,226), 2)
            
            # Save the image with the bounding boxes
            cv2.imwrite(image_filename, rider_crop)
            print(f"License Plate Image saved as {image_filename}")