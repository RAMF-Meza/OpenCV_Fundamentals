# Assignment 1
# Import modules
import cv2
import matplotlib.pyplot as plt
import matplotlib
#from dataPath import DATA_PATH

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)

imgPath = 'image.png'

# Read image
img = cv2.imread(imgPath)
plt.imshow(img)
plt.show(block=False)

# Step 2: Detect QR Code in the Image

# Create a QRCodeDetector Object
# Variable name should be qrDecoder

qrDecoder = cv2.QRCodeDetector()

# Detect QR Code in the Image
# Output should be stored in
# opencvData, bbox, rectifiedImage
# in the same order

opencvData, bbox, recifiedImage = qrDecoder.detectAndDecode(img)

# Check if a QR Code has been detected
if opencvData != None:
    print("QR Code Detected")
else:
    print("QR Code NOT Detected")

# STEP 3: DRAW BOUNDING BOX AROUND THE DETECTED QR CODE
n = len(bbox)
print(n)
print(bbox[2][0])

# Draw the bounding box

def display(im, bbox):
    pt1 = tuple(int(i) for i in bbox[0][0])
    pt2 = tuple(int(i) for i in bbox[2][0])
    cv2.rectangle(im, pt1, pt2, (255,0,0), 3)
    
    plt.imshow(im[...,::-1])
    plt.show()

display(img, bbox)

# SETP 4: PRINT THE DECODED TEXT
print("QR Code Detected!")
print("Decoded Data: {}".format(opencvData))

# STEP 5: SAVE AND DISPLAY THE RESULT

# Write the result image
resultImagePath = "QRCode-Output.png"
cv2.imwrite(resultImagePath, img)
img = cv2.imread(resultImagePath)

# Display the result image
plt.imshow(img)

# OpenCV uses BGR whereas Matplotlib uses RGB format
# So convert the BGR image to RGB image
# And display the correct image

plt.imshow(img[:,:,::-1])
