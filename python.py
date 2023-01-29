import numpy as np
import cv2
from PIL import Image

# Open image and convert to RGB
im = Image.open("image.jpg").convert("RGB")

# Convert image to numpy array and grayscale
np_im = np.array(im)
gray = cv2.cvtColor(np_im, cv2.COLOR_RGB2GRAY)

# Define range of color in HSV
lower_color = np.array([0,50,50])
upper_color = np.array([10,255,255])

# Threshold the image to get only the specified color
hsv = cv2.cvtColor(np_im, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(hsv, lower_color, upper_color)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
max_area = 0
largest_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        largest_contour = contour

# Crop the image to the shape of the largest contour
if largest_contour is not None:
    x,y,w,h = cv2.boundingRect(largest_contour)
    crop_im = im.crop((x, y, x+w, y+h))
    crop_im.show()
else:
    print("No area found with the specified color.")
