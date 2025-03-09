import cv2
import matplotlib.pyplot as plt
import numpy as np


def cartoonify(image, arguments=0):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Median blur
    gray = cv2.medianBlur(gray, 5)
    # Detect the edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9,9)
    # Apply bilateral filter to smooth the image
    color = cv2.bilateralFilter(image, 9, 300, 300)
    # Combine edges and colored image
    cartoonImage = cv2.bitwise_and(color, color, mask = edges)

    return cartoonImage

def pencilSketch(image, arguments=0):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert the grayscale image
    inverted_gray = 255 - gray
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted_gray, (21,21), 0)
    # Revert to the original scale
    inverted_blurred = 255 - blurred
    # Create the pencil effect
    pencilSketchImage = cv2.divide(gray, inverted_blurred, scale = 256.0)
    pencilSketchImage = cv2.cvtColor(pencilSketchImage, cv2.COLOR_GRAY2BGR)

    return pencilSketchImage

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image, kernel)
    return sepia

imagePath = "image.jpg"
image = cv2.imread(imagePath)

cartoonImage = cartoonify(image)
pencilSketchImage = pencilSketch(image)
sepia = apply_sepia(image)

plt.figure(figsize=[20,10])
plt.subplot(141);plt.imshow(image[:,:,::-1]);
plt.subplot(142);plt.imshow(cartoonImage[:,:,::-1]);
plt.subplot(143);plt.imshow(pencilSketchImage[:,:,::-1]);
plt.subplot(144);plt.imshow(sepia[:,:,::-1]);