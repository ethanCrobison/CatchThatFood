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



##--------------- Initialize --------------------------------------------

# openCV
cap = cv2.VideoCapture(0)   # instantiate webcam feed

ret, frame = cap.read()     # check that frame is valid
while frame is None:
    ret, frame = cap.read()

cascPath = 'haarcascades/haarcascade_frontalface_default.xml'   # instantiate face detection
faceCascade = cv2.CascadeClassifier(cascPath)

# variables
WINDOWHEIGHT, width, depth = frame.shape  # image dimensions

points = 0  # score

items = []
itemCounter = 1
newItemAt = 20

RED     = (  0,   0, 255)
GREEN   = (  0, 255,   0)
BLUE    = (255,   0,   0)
##--------------- Main Loop ---------------------------------------------
while True:
    ret, frame = cap.read()

    # face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # draw a rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for i in items:
            tx = i['x']
            ty = i['y']
            th = i['height']
            tw = i['width']

            if tx+tw > x and tx < x+w and ty > y and ty+th < y+h:
                i['color'] = GREEN
            elif i['color'] != RED:
                i['color'] = RED

    
    # game stuff
    itemCounter += 1
    if itemCounter > newItemAt:
        itemCounter = 1
        newItem = {
            'x': 200+random.randint(-150, 150),
            'y': 10,
            'width': 10,
            'height': 10,
            'color': RED
            }
        items.append(newItem)

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

        i['y'] += 5
        cv2.rectangle(frame, (tx,ty),(tx+tw,ty+th),co,2)


    # update frame
    cv2.imshow('game_window', frame)


    # check for quitting
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break

##-----------------------------------------------------------------------


##  mission critical
cap.release()
cv2.destroyAllWindows()
