##
## CatchThatFood!
##
## Created by Ethan Robison & Michael Wang
##
## EECS 332 Digital Image Analysis Fall 2014
## Professor Ying Wu
##


import numpy as np
import cv2
##import random, time

##  pass 0 or -1 to use default cam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

##  create window ahead of time
cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)

##  check that frame is valid
while frame is None:
    ret, frame = cap.read()


##--------------- Main Loop ---------------------------------------------
while True:
##  read frame  
    ret, frame = cap.read()

##  operate on frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

##  display frame
    cv2.imshow('Gray', gray)



##  check for quitting
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break

##-----------------------------------------------------------------------

cv2.waitKey(0)
##  mission critical
cap.release()
cv2.destroyAllWindows()
