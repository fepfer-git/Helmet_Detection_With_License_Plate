from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO('license_plate_2.pt')

# Load and perform inference on the image
image_path = 'test5.jpg'
results = model(image_path)

# Print class names, bounding boxes, and confidence scores
for result in results[0].boxes.data.tolist():  # Access the first result set
    x1, y1, x2, y2, confidence, class_id = result
    class_name = results[0].names[int(class_id)]
    print(f"Class: {class_name}, ClassId: {class_id}, Confidence: {confidence:.2f}, BBox: ({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})")

# Plot the image with bounding boxes and labels
annotated_image = results[0].plot()  # Plot the boxes and labels on the image

# Display the image
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()
