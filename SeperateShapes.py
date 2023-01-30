import cv2
import numpy as np

# Load the image
image = cv2.imread("shapes1.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to pre-process the image
_, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through all contours
for i, contour in enumerate(contours):
    # Get approximate polyggon for contour
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    
    # Detect the shape of the contour
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
            shape = "Square"
        else:
            shape = "Rectangle"
    elif len(approx) == 5:
        shape = "Pentagon"
    elif len(approx) == 6:
        shape = "Hexagon"
    else:
        shape = "Circle"
        
    # Crop the shape from the image
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    crop = cv2.bitwise_and(image, image, mask=mask)
    
    # Save the cropped image
    cv2.imwrite("SeperatedImages/{}_{}.jpg".format(shape, i), crop)
