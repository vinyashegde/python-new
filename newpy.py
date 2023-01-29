import numpy as np
import cv2
from PIL import Image
from tkinter import filedialog
from tkinter import Tk

root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

im = Image.open(file_path).convert("RGB")
np_im = np.array(im)
gray = cv2.cvtColor(np_im, cv2.COLOR_RGB2GRAY)

lower_color = np.array([0,50,50])
upper_color = np.array([10,255,255])

hsv = cv2.cvtColor(np_im, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(hsv, lower_color, upper_color)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
largest_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
if area > max_area:
    max_area = area
    largest_contour = contour

if largest_contour is not None:
    x,y,w,h = cv2.boundingRect(largest_contour)
    crop_im = im.crop((x, y, x+w, y+h))
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
    crop_im.save(save_path)
else:
    print("No area found with the specified color.")