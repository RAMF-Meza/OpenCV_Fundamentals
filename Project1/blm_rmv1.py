import cv2
import numpy as np

image = cv2.imread('blemish.png')


def remove_blemish(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        blemish_size = 10

        # Region of Interest
        x1, y1 = max(0, x - blemish_size), max(0, y-blemish_size)
        x2, y2 = min(image.shape[1], x + blemish_size), min(image.shape[0], y+blemish_size)
        
        # Find patch
        patch_x, patch_y = x + 2 * blemish_size, y
        patch_x1, patch_y1 = max(0, patch_x - blemish_size), max(0, patch_y - blemish_size)
        patch_x2, patch_y2 = min(image.shape[1], patch_x + blemish_size), min(image.shape[0], patch_y + blemish_size)
        patch_roi = image[patch_y1:patch_y2, patch_x1:patch_x2]

        # Create a mask for seamless cloning
        mask = 255 * np.ones(patch_roi.shape, patch_roi.dtype)

        # Apply seamless cloning
        center = (x, y)
        image_clone = cv2.seamlessClone(patch_roi, image, mask, center, cv2.NORMAL_CLONE)

        # Update the image with the cloned result
        image[y1:y2, x1:x2] = image_clone[y1:y2, x1:x2]


cv2.namedWindow('Blemish Removal')
cv2.setMouseCallback('Blemish Removal', remove_blemish)

while True:
    cv2.imshow('Blemish Removal', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()