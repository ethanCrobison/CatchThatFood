##
## CatchThatFood!
##
## Created by Ethan Robison & Michael Wang
##
## EECS 332 Digital Image Analysis Fall 2014
## Professor Ying Wu
##
import cv2
import pygame, random, sys
from operator import itemgetter

##--------------- Functions ---------------------------------------------
def within(outer, inner):
    print 'Placeholder'


##--------------- Initialize --------------------------------------------

# openCV
cap = cv2.VideoCapture(0)   # instantiate webcam feed

ret, frame = cap.read()     # check that frame is valid
while frame is None:
    ret, frame = cap.read()

faceCascadePath = 'haarcascades/haarcascade_frontalface_default.xml'   # instantiate face detection
faceCascade = cv2.CascadeClassifier(faceCascadePath)
mouthCascadePath = 'haarcascades/haarcascade_mcs_mouth.xml'
mouthCascade = cv2.CascadeClassifier(mouthCascadePath)

# variables
WINDOWHEIGHT, WINDOWWIDTH, depth = frame.shape  # image dimensions
XHALF = WINDOWWIDTH / 2
YHALF = WINDOWHEIGHT / 2
points = 0  # score

items = []
itemCounter = 1
newItemAt = 20

traps = []
trapCounter = -40
newTrapAt = 30

# RGB CODES
RED     = (  0,   0, 255)
GREEN   = (  0, 255,   0)
BLUE    = (255,   0,   0)
##--------------- Main Loop ---------------------------------------------
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(100, 100),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
    	# cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    	roi_gray = gray[y:y+h, x:x+w]
    	roi_color = frame[y:y+h, x:x+w]

    	mouths = mouthCascade.detectMultiScale(
    		roi_gray,
	    	scaleFactor=1.1,
	    	minNeighbors=5,
	    	minSize=(50, 30),
	    	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    	)

    	if list(mouths):
	    	y_values = zip(*mouths)[1]
	    	max_index = y_values.index(max(y_values))
	    	lowest_mouth = mouths[max_index]

	    	mx = lowest_mouth[0]
	    	my = lowest_mouth[1]
	    	mw = lowest_mouth[2]
	    	mh = lowest_mouth[3]
	    	cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)

    # draw a rectangle around faces
    for (x,y,w,h) in faces:
        for i in items:
            tx = i['x']
            ty = i['y']
            th = i['height']
            tw = i['width']

            if tx+tw > x and tx < x+w and ty > y and ty+th < y+h:
                items.remove(i)
                points += 10

    
    # add items, etc.
    itemCounter += 1
    if itemCounter > newItemAt:
        itemCounter = 1
        newItem = {
            'x': XHALF + random.randint(-XHALF + 50, XHALF - 50),
            'y': 10,
            'width': 10,
            'height': 10,
            'color': RED
            }
        items.append(newItem)

    trapCounter += 1
    if trapCounter > newTrapAt:
        trapCounter = 1
        newTrap = {
            'x': 10,
            'y': YHALF + random.randint(-YHALF + 50, YHALF -50),
            'width': 10,
            'height': 10,
            'color': BLUE,
            'speed': random.randint(5, 15)
            }
        traps.append(newTrap)

    # handle items
    for i in items:
        tx = i['x']
        ty = i['y']
        th = i['height']
        tw = i['width']
        co = i['color']
        if ty + th >= WINDOWHEIGHT:
            items.remove(i)
            continue
        i['y'] += 10
        cv2.rectangle(frame, (tx,ty),(tx+tw,ty+th),co,2)

    for t in traps:
        tx = t['x']
        ty = t['y']
        th = t['height']
        tw = t['width']
        co = t['color']
        if tx + tw >= WINDOWWIDTH:
            traps.remove(t)
            continue
        t['x'] += t['speed']
        cv2.rectangle(frame, (tx,ty),(tx+tw,ty+th),co,2)

    # handle score
    cv2.putText(frame,"Points: %d" %(points),(10,55),cv2.FONT_HERSHEY_COMPLEX,2,255)
    
    # update frame
    cv2.imshow('game_window', frame)

    # check for quitting
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break

##-----------------------------------------------------------------------


##  mission critical
cap.release()
cv2.destroyAllWindows()
