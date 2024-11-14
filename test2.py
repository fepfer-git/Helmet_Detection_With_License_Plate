import os
import cv2
from ultralytics import YOLO

    # Initialize YOLO model
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

            if confidence > 0.1 and cls == 2:  # Class 2: Motorbike Rider
                x1, y1, x2, y2 = map(int, box)
                rider_boxes.append((x1, y1, x2, y2))

    print(f"Number of riders detected: {len(rider_boxes)}")
    print(f"Rider boxes: {rider_boxes}")
    # Dictionary to store rider boxes and their corresponding helmet and license plate
    rider_info = {}

    for idx, rider_box in enumerate(rider_boxes):
        x1, y1, x2, y2 = rider_box
        rider_crop = frame[y1:y2, x1:x2]

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
                        
        #check again with another model
        for result in helmet_detection_results:
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

                    if cls == 0:  # Class 0: Helmet
                        helmet_found = True
        
        
        bx1, by1, bx2, by2 = license_plate
        license_plate_crop = frame[by1:by2, bx1:bx2]

        rider_info[idx] = {
            "helmet_found": helmet_found,
            "license_plate": license_plate,
            "rider_box": rider_box
        }
        
        if helmet_found == False and license_plate is not None:
            image_filename = os.path.join(output_folder, f"Rider_{idx}.jpg")
            print(f"Rider {idx} - Helmet Found: {helmet_found}")
            
            # Draw the bounding box around the license plate
            bx1, by1, bx2, by2 = license_plate
            cv2.rectangle(rider_crop, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), (0, 0, 255), 1)
            cv2.putText(rider_crop, "License_plate", (bx1 - x1, by1 - y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            
            # Save the image with the bounding boxes
            cv2.imwrite(image_filename, rider_crop)
            
            # Display the image with the bounding boxes
            cv2.imshow(f"License Plate of Rider {idx} without Helmet", rider_crop)
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()