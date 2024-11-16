import cv2
from matplotlib import pyplot as plt
import easyocr

# Read the image
img = cv2.imread('license_test.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.show()

# Apply bilateral filter to reduce noise
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

# Use EasyOCR to read text
reader = easyocr.Reader(['en'])
results = reader.readtext(img)

license_plate_text = ''
if results:
    for i in results :
        license_plate_text = license_plate_text + ' ' + i[1]
    
print(license_plate_text)
