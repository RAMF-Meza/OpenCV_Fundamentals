# Assignment 2
import cv2
import math
import matplotlib

leftUpperCorner = []
rightLowerCorner = []
color = (255,255,0)
width = 2

def drawRectangle(action, x,y, flags, userdata):
    
    global leftUpperCorner, rightLowerCorner

    # When pressing the button
    if action == cv2.EVENT_LBUTTONDOWN:
        leftUpperCorner = [(x,y)]
    # Whe mouse is released 
    elif action == cv2.EVENT_LBUTTONUP:
        rightLowerCorner = [(x,y)]
    
        # Draw the rectangle
        cv2.rectangle(source, leftUpperCorner[0],rightLowerCorner[0], color, width)
        cv2.imshow("Window", source)
        new_img = source[leftUpperCorner[0][1]:rightLowerCorner[0][1], leftUpperCorner[0][0]:rightLowerCorner[0][0],:]
        cv2.imwrite("face.jpg", new_img)
    #cv.rectangle(img,(384,0),(510,128),(0,255,0),3)

source = cv2.imread("sample2.jpg",1)
#print(source.shape)
# Making a dummy image
dummy = source.copy()
cv2.namedWindow("Window")

cv2.setMouseCallback("Window", drawRectangle)
k = 0

while k != 27:
    cv2.imshow("Window", source)
    cv2.putText(source, '''Choose the left upper corner and drag''', (10,500), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, width)
    k = cv2.waitKey(20) & 0xFF
    if k == 99:
        source = dummy.copy()

cv2.destroyAllWindows()