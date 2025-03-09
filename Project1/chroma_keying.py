# OpenCV Week 6 Project Chroma Keying
import cv2
import numpy as np

video_path = "02_fundamentals_opencv\week6_python\chroma_keying\greenscreen-demo.mp4"
background_path = "02_fundamentals_opencv\week4-python\data\\videos\\focus-test.mp4"

# Control variables
selected_color = None
tolerance = 30
softness = 10
color_cast = 0

cap = cv2.VideoCapture(video_path)
bg_cap = cv2.VideoCapture(background_path)

def nothing():
    pass

def select_color(event, x, y, flags, param):
    global selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_color = frame[y,x]
# GUI
panel = np.zeros([200,700], np.uint8)
#cv2.namedWindow("panel")
cv2.namedWindow("Chroma Key")
cv2.createTrackbar("Tolerance","Chroma Key", tolerance, 100, nothing)
cv2.createTrackbar("Softness","Chroma Key", softness, 50, nothing)
cv2.createTrackbar("Color Cast","Chroma Key", color_cast, 100, nothing)
cv2.setMouseCallback("Chroma Key", select_color)

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("Chroma_Key.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

def switch_bg(frame, bg_frame):
    global selected_color, tolerance, softness, color_cast
    if selected_color is None:
        return frame
    
    # Convert color space to hsv
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_color = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]

    # Define HSV range for chroma key based on tolerance
    lower = np.array([max(hsv_color[0] - tolerance, 0), 50, 50])
    upper = np.array([min(hsv_color[0] + tolerance, 179), 255, 255])

    # Create a mask for the green screen
    mask = cv2.inRange(hsv_frame, lower, upper)

    # Apply softness (blur the mask)
    mask = cv2.GaussianBlur(mask, (softness * 2 + 1, softness * 2 + 1), 0)

    # Remove color cast by subtracting green from the subject
    cast_removed = frame.copy()
    if color_cast > 0:
        cast_removed[:, :, 1] = np.clip(cast_removed[:, :, 1] - color_cast, 0, 255)

    # Replace the green screen with the background
    bg_resized = cv2.resize(bg_frame, (frame.shape[1], frame.shape[0]))
    fg_mask = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(cast_removed, cast_removed, mask=fg_mask)
    bg = cv2.bitwise_and(bg_resized, bg_resized, mask=mask)
    output = cv2.add(fg, bg)

    return output




while True:
    ret, frame = cap.read(0)

    ret_bg, bg_frame = bg_cap.read()
    if not ret_bg:
        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, bg_frame = bg_cap.read()
    
    tolerance = cv2.getTrackbarPos("Tolerance","Chroma Key",)
    softness = cv2.getTrackbarPos("Softness","Chroma Key",)
    color_cast = cv2.getTrackbarPos("Color Cast","Chroma Key",)

    # Switch the background when the color is selected
    output_frame = switch_bg(frame, bg_frame)

    # Display the new video
    cv2.imshow("Chroma Key", output_frame)

    # Write the new video
    out.write(output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
bg_cap.release()
out.release()
cv2.destroyAllWindows()