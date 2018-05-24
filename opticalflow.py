#!/usr/bin/env python3
import numpy as np
import cv2

# Draws optic flow visualisation on image using a given step size for the line glyphs that show the flow vectors on the image
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:      
	    cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

# Define video capture object
cap = cv2.VideoCapture('video.avi')

# Take first frame and convert to gray scale image
ret, frame = cap.read()
prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Play until the user decides to stop
while True:
    # Get next frame
    ret, frame1 = cap.read()
    next = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    
    # Calculate the dense optical flow
    flow = cv2.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Display final frame
    cv2.imshow('hi', draw_flow(next, flow))
    k = cv2.waitKey(30) & 0xff

    # Exit if the user presses ESC
    if k == 27:
        break

cv2.destroyAllWindows()