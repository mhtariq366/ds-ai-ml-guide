# import the opencv-python library
import cv2

"""
to read an image in opencv, use the imread function of cv2 module.
the first parameter is the image path, the second parameter decides if image is to be
read in black and white format then 0 if in color format then 1.
"""

img_arr = cv2.imread('image_01.png', 0)

#   print the image in matrix
print(img_arr)