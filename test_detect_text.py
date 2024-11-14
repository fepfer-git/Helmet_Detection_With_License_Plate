import cv2
import pytesseract

def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding for better results
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def detect_license_plate(image):
    # Placeholder function for detecting license plate.
    # You can use Haar Cascades, YOLO, or another method for license plate detection.
    # Assuming this function returns the coordinates of the license plate
    # Here, it will just return the full image as a placeholder.
    return image  # Replace this with detected plate region

def read_license_plate(image_path):
    # Preprocess image
    processed_image = preprocess_image(image_path)
    
    # Detect the license plate region (coordinates could also be used to crop)
    plate_region = detect_license_plate(processed_image)
    
    # Perform OCR on license plate region
    config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(plate_region, config=config)
    
    # Post-process to clean up unwanted characters
    license_plate = "".join([c for c in text if c.isalnum()])
    
    return license_plate

# Usage
license_plate_text = read_license_plate('license_test.jpg')
print("Detected License Plate:", license_plate_text)
