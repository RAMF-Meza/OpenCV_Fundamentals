# Assignment 4

"""
The main steps that you can follow to solve this assignment are:

1. Read the image.
2. Convert it to grayscale and split the image into the 3 (Red, Green and Blue) channels. Decide which of the above 4 images you want to use in further steps and provide reason for the same.
3. Use thresholding and/or morphological operations to arrive at a final binary image.
4. Use simple blob detector to count the number of coins present in the image.
5. Use contour detection to count the number of coins present in the image.
6. Use CCA to count the number of coins present in the image.
"""

# STEP 1: READ IMAGE
import cv2
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Image path
imagePath = "data/CoinsA.png"
# Read image
# Store it in the variable image
image = cv2.imread(imagePath,)
imageCopy = image.copy()
plt.imshow(image[:,:,::-1])
plt.title("Original Image")

cv2.imshow("image", image[:,:,::-1])

# STEP 2: CONVERT IMAGE TO GRAYSCALE

imageGray = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(image[:,:,::-1])
plt.title("Original Image")
plt.subplot(122)
plt.imshow(imageGray)
plt.title("Grayscale Image")

# STEP 2: SPLIT IMAGE INTO R G B CHANNELS

# Split cell into channels
# Store them in variables imageB, imageG, imageR
imgRGB = cv2.cvtColor(imageCopy,cv2.COLOR_BGR2RGB)
imageR = imgRGB[:,:,0]
imageG = imgRGB[:,:,1]
imageB = imgRGB[:,:,2]

plt.figure(figsize=(20,12))
plt.subplot(141)
plt.imshow(image[:,:,::-1])
plt.title("Original Image")
plt.subplot(142)
plt.imshow(imageB)
plt.title("Blue Channel")
plt.subplot(143)
plt.imshow(imageG)
plt.title("Green Channel")
plt.subplot(144)
plt.imshow(imageR)
plt.title("Red Channel")

# STEP 3: PERFORM THRESHOLDING

thresh = 200
maxValue = 255

images = [imageGray, imageR, imageG, imageB]
for i in range(len(images)):
    src = images[i]
    th, dst_bin = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)
    th, dst_bin_inv = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY_INV)
    th, dst_trunc = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TRUNC)
    th, dst_to_zero = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TOZERO)
    th, dst_to_zero_inv = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TOZERO_INV)

    print("Threshold Value = {}, Max Value = {}".format(thresh, maxValue))
    plt.figure(figsize=[20,12])
    plt.subplot(231);plt.imshow(src, cmap='gray', vmin=0, vmax=255);plt.title("Original Image")
    plt.subplot(232);plt.imshow(dst_bin, cmap='gray', vmin=0, vmax=255);plt.title("Threshold Binary")
    plt.subplot(233);plt.imshow(dst_bin_inv, cmap='gray', vmin=0, vmax=255);plt.title("Threshold Binary Inverse")
    plt.subplot(234);plt.imshow(dst_trunc, cmap='gray', vmin=0, vmax=255);plt.title("Threshold Truncate")
    plt.subplot(235);plt.imshow(dst_to_zero, cmap='gray', vmin=0, vmax=255);plt.title("Threshold To Zero")
    plt.subplot(236);plt.imshow(dst_to_zero_inv, cmap='gray', vmin=0, vmax=255);plt.title("Threshold To Zero Inverse")
plt.show()

# STEP 3.2: PERFORM MORPHOLOGICAL OPERATIONS

# Experimenting with threshold on ImageG
th, dst_trunc = cv2.threshold(imageG, thresh, maxValue, cv2.THRESH_TRUNC)
th, dst_to_zero_inv = cv2.threshold(dst_trunc, thresh, maxValue, cv2.THRESH_TOZERO_INV)
th, dst_bin_inv = cv2.threshold(dst_to_zero_inv, 60, maxValue, cv2.THRESH_BINARY_INV)
th, dst_to_zero = cv2.threshold(dst_bin_inv, 200, maxValue, cv2.THRESH_TOZERO)

plt.figure(figsize=[20,12])
plt.subplot(231);plt.imshow(imageG, cmap='gray', vmin=0, vmax=255);plt.title("Green Channel")
plt.subplot(232);plt.imshow(dst_trunc, cmap='gray', vmin=0, vmax=255);plt.title("Threshold Truncate")
plt.subplot(233);plt.imshow(dst_to_zero_inv, cmap='gray', vmin=0, vmax=255);plt.title("Threshold To Zero Inverse")
plt.subplot(234);plt.imshow(dst_bin_inv, cmap='gray', vmin=0, vmax=255);plt.title("Threshold Binary Inverse")
plt.subplot(235);plt.imshow(dst_to_zero, cmap='gray', vmin=0, vmax=255);plt.title("Threshold")
plt.show()

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
dilated = cv2.dilate(dst_bin_inv, element, iterations=1)
eroted = cv2.erode(dilated, element, iterations=3)

closingSize = 5
# Selecting an elliptical kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * closingSize + 1, 2 * closingSize + 1),(closingSize,closingSize))
imageMorphClosed = cv2.morphologyEx(eroted,cv2.MORPH_CLOSE, element, iterations=3)

plt.figure(figsize=[20,12])
plt.title("Morphological Operations")
plt.imshow(imageMorphClosed)
plt.show()

dilated = cv2.dilate(imageMorphClosed, element, iterations=2)
opened = cv2.morphologyEx(dilated,cv2.MORPH_OPEN, element, iterations=1)
closed = cv2.morphologyEx(opened,cv2.MORPH_CLOSE, element, iterations=1)
eroted = cv2.erode(closed, element, iterations=2)

plt.figure(figsize=[20,12])
plt.title("New")
plt.imshow(eroted)
plt.show()

# STEP 4: CREATE SIMPLE BLOB DETECTOR

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.8

# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)

# STEP 4.2: DETECT COINS

# Detect blobs
image_to_blob = eroted

keypoints = detector.detect(image_to_blob)

im = cv2.cvtColor(image_to_blob, cv2.COLOR_GRAY2BGR)

# Print number of coins detected
print("Number of coins: ",len(keypoints))

# STEP 4.3: DISPLAY THE DETECTED COINS ON ORIGINAL IMAGE

# Mark coins using image annotation concepts we have studied so far
im = cv2.cvtColor(image_to_blob, cv2.COLOR_GRAY2BGR)

for k in keypoints:
    x,y = k.pt
    x=int(round(x))
    y=int(round(y))
    # Mark center in BLACK
    cv2.circle(image,(x,y),5,(255,0,0),-1)
    # Get radius of blob
    diameter = k.size
    radius = int(round(diameter/2))
    # Mark blob in RED
    cv2.circle(image,(x,y),radius,(0,255,0),3)
    
# Let's see what image we are dealing with
print("Number of coins: ",len(keypoints))
plt.figure(figsize=[20,12])
plt.title("Blob Detector on Original Image")
plt.imshow(image[:,:,::-1])
plt.show()

# STEP 4.4: PERFORM CONNECTED COMPONENT ANALYSIS
def displayConnectedComponents(im):
    imLabels = im
    # The following line finds the min and max pixel values
    # and their locations in an image.
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imLabels)
    # Normalize the image so the min value is 0 and max value is 255.
    imLabels = 255 * (imLabels - minVal)/(maxVal-minVal)
    # Convert image to 8-bits unsigned type
    imLabels = np.uint8(imLabels)
    # Apply a color map
    imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)
    # Display colormapped labels
    plt.imshow(imColorMap[:,:,::-1])

# Threshold Image
th, imThresh = cv2.threshold(image_to_blob, 127, 255, cv2.THRESH_BINARY_INV)
plt.imshow(imThresh)

# Find connected components
_, imLabels = cv2.connectedComponents(imThresh)
plt.imshow(imLabels)

# Print number of connected components detected
nComponents = imLabels.max()
print("Number of CC: ", nComponents)

displayRows = np.ceil(nComponents/3.0)
plt.figure(figsize=[20,12])
for i in range(nComponents+1):
    plt.subplot(displayRows,4,i+1)
    plt.imshow(imLabels==i)
    if i == 0:
        plt.title("Background, Component ID : {}".format(i))
    else:
        plt.title("Component ID : {}".format(i))

(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imLabels)

# Normalize the image so that the min value is 0 and max value is 255.
imLabels = 255 * (imLabels - minVal)/(maxVal-minVal)

# Convert image to 8-bits unsigned type
imLabels = np.uint8(imLabels)

# Apply a color map
imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)
plt.imshow(imColorMap[:,:,::-1])

# Display connected components using displayConnectedComponents
# function
displayConnectedComponents(imThresh)

# STEP 4.5: DETECT COINS USING CONTOUR DETECTION

# Image path
imagePath = "data/CoinsA.png"
# Read image
# Store it in the variable image
image = cv2.imread(imagePath,)
imageCopy = image.copy()

# Find all contours in the image
#image2 = image.copy()
image2 = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)

contours, hierarchy = cv2.findContours(imThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours found = {}".format(len(contours)))
print("\nHierarchy : \n{}".format(hierarchy))

plt.imshow(image2)

# Draw all contours
cv2.drawContours(image2, contours, -1, (0,255,0), 3);
plt.figure(figsize=[20,12])
plt.title("Contours")
plt.imshow(image2)
plt.show()

# Remove the inner contours
contours, hierarchy = cv2.findContours(imThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found = {}".format(len(contours)))
# Draw all contours
cv2.drawContours(image2, contours, -1, (255,255,0), 3);
plt.figure(figsize=[20,12])
plt.title("Contours")
plt.imshow(image2)
plt.show()

# Print area and perimeter of all contours
for index,cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print("Contour #{} has area = {} and perimeter = {}".format(index+1,area,perimeter))

# Fit circles on coins
image3 = imageCopy.copy()
for cnt in contours:
    # Fit a circle
    ((x,y),radius) = cv2.minEnclosingCircle(cnt)
    cv2.circle(image, (int(x),int(y)), int(round(radius)), (0,125,255), 3)
    
plt.imshow(image[:,:,::-1])

""" ASSIGNMENT PART - B"""