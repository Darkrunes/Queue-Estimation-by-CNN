#!/usr/bin/env python3
import numpy as np
import cv2
import sys
import time

farneback_params = {
    'pyr_scale':0.5,
    'levels':3,
    'winsize':15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma':1.2,
    'flags':0
    } 

def draw_hsv(flow):
    h, w = flow.shape[:2]
    # Compute the magnitude and angle of 2D vectors
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((h, w, 3), np.uint8)
    # Corresponds to hue
    hsv[...,0] = ang*(180/np.pi/2)
    # Corresponds to saturation
    hsv[...,1] = 255
    # Corresponds to value
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

# Define video capture object
cap = cv2.VideoCapture('pedtest.mp4')

# Take first frame and convert to gray scale image
ret, first_frame = cap.read()
prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Play until the user decides to stop
while True:
    # Get next frame and convert to gray scale
    ret, next_frame = cap.read()
    curr_frame = next_frame.copy()
    next = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
    # Calculate the dense optical flow between two frames
    flow = cv2.calcOpticalFlowFarneback(prev,next,None,**farneback_params)
    prev = next
    #cv2.imshow('Hi', draw_hsv(flow))
    hsv = cv2.cvtColor(draw_hsv(flow), cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5), np.uint8)
    #dilation = cv2.dilate(hsv, kernel, iterations=2)
    thresh = cv2.threshold(hsv, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    edg_er = cv2.erode(thresh, kernel, iterations = 3)
    blurred_frame = cv2.GaussianBlur(thresh, (5, 5), 0)
    _, contours, _  = cv2.findContours(blurred_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	

    # Computes the bounding box for the contour, and draws it on the frame
    for contour in contours:
        #area = cv2.contourArea(contour)
        #if(area>300):
        x,y,w,h = cv2.boundingRect(contour)
        if w > 100 and h > 100 and w < 900 and h < 680:
            cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(curr_frame,str(time.time()), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
    # Displays current frame
    #cv2.imshow('hello', curr_frame)
    #cv2.drawContours(curr_frame, contours, -1, (0,255,0), 3)
    cv2.imshow('hello', curr_frame)
    k = cv2.waitKey(30) & 0xff
    # Exit if the user presses ESC
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
