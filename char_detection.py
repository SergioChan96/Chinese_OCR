import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

newspaper = cv2.imread('resources/newspaper.jpg')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def rgb2binary(img, threshold):
    # get dimensions
    height, width = img.shape

    new_img = img.copy()

    # apply filter
    for i in range(1, height-1):
        for j in range(1, width-1):
            if img[i,j] < threshold:
                new_img[i,j] = 0
            else:
                new_img[i,j] = 255
    
    return new_img

def averaging_filter(img):
    # get dimensions
    height, width = img.shape

    new_img = img.copy()

    # apply filter
    for i in range(1, height-1):
        for j in range(1, width-1):
            new_img[i,j] = (img[i-1,j-1] + img[i-1,j] + img[i-1,j+1] +
                            img[i,j-1]   + img[i,j]   + img[i,j+1] +
                            img[i+1,j-1] + img[i+1,j] + img[i+1,j+1]) // 9
    
    return new_img

def median_filter(img):
    # get dimensions
    height, width = img.shape

    new_img = img.copy()

    # apply filter
    for i in range(1, height-1):
        for j in range(1, width-1):
            new_img[i,j] = np.median([img[i-1,j-1], img[i-1,j], img[i-1,j+1],
                                      img[i,j-1],   img[i,j],   img[i,j+1],
                                      img[i+1,j-1], img[i+1,j], img[i+1,j+1]])
    
    return new_img

def sharpen_edges(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    # get dimensions
    height, width = img.shape

    new_img = img.copy()

    # apply filter
    for i in range(1, height-1):
        for j in range(1, width-1):
            new_img[i,j] = (img[i-1,j-1] * kernel[0,0] + img[i-1,j] * kernel[0,1] + img[i-1,j+1] * kernel[0,2] +
                            img[i,j-1]   * kernel[1,0] + img[i,j]   * kernel[1,1] + img[i,j+1]   * kernel[1,2] +
                            img[i+1,j-1] * kernel[2,0] + img[i+1,j] * kernel[2,1] + img[i+1,j+1] * kernel[2,2])
            
    return new_img

def preprocess(img):
    # convert to grayscale
    img = rgb2gray(img)

    # invert colors
    img = 255 - img

    # apply averaging filter
    img = averaging_filter(img)

    # apply median filter
    # img = median_filter(img)

    # sharpen the edges
    img = sharpen_edges(img)

    # convert to binary
    img = rgb2binary(img, 128)

    return img

img = preprocess(newspaper)

plt.imshow(img, cmap='gray')
plt.show()

img = img.astype('uint8')

print(img)

# Find contours in the binary image
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour
for contour in contours:
    # Get bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)

    # You can now filter the contours based on the properties of the bounding box
    # For example, to filter out bounding boxes that are too small:
    if w > 10 and h > 10:
        # Draw the bounding box on the image (just for visualization)
        cv2.rectangle(img, (x, y), (x+w, y+h), 255, 2)

print('Number of contours found: {}'.format(len(contours)))

# Display the image
plt.imshow(img, cmap='gray')
plt.show()
