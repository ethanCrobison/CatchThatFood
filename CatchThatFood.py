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

def outside(item, screenDim):
    return item['x']+item['w'] >= screenDim[0] or item['y']+item['h'] >= screenDim[1]
        


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

##--------------- Game Stuff --------------------------------------------
# constants
WINDOWHEIGHT, WINDOWWIDTH, depth = frame.shape  # image dimensions
DIMS = [WINDOWWIDTH, WINDOWHEIGHT]

XHALF = WINDOWWIDTH / 2
YHALF = WINDOWHEIGHT / 2

RED     = (  0,   0, 255)
GREEN   = (  0, 255,   0)
BLUE    = (255,   0,   0)

# variables
points = 0  # score
health = 5

items = []

pointC      = 0     # good item counter
NEWPOINT    = 20    # interval

trapC       = -60   # trap item counter
NEWTRAP     = 30    # interval

healC       = -100  # heal item counter
NEWHEAL     = 50    # interval

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

	    	for i in items[:]:
		        tx = i['x']
		        ty = i['y']
		        th = i['h']
		        tw = i['w']

		        if tx+tw > mx+x and tx < mx+x+mw and ty > my+y and ty+th < my+y+mh:
		            items.remove(i)
		            points += i['po']
    
    # add items, etc.
    pointC  += 1
    trapC   += 1
    healC   += 1

    if pointC > NEWPOINT:
        pointC = 1
        newItem = {
            'x': random.randint(20, WINDOWWIDTH - 20),    # x coord
            'y': 10,    # y coord
            'xs': 0,    # x speed
            'ys': 10,   # y speed
            'w': 10,    # width
            'h': 10,    # height
            'co': RED,  # color
            'po': 10,   # points value
            't': 0     
            }
        items.append(newItem)
        
    if trapC > NEWTRAP:
        trapC = 1
        newItem = {
            'x': 10,    # x coord
            'y': random.randint(20, WINDOWHEIGHT - 20),    # y coord
            'xs': random.randint(5, 10), # x speed
            'ys': 0,    # y speed
            'w': 10,    # width
            'h': 10,    # height
            'co': BLUE, # color
            'po': 0,    # points value
            't': 0     
            }
        items.append(newItem)
        
    if healC > NEWHEAL:
        healC = 1
        newItem = {
            'x': random.randint(50, WINDOWWIDTH - 50),    # x coord
            'y': random.randint(50, WINDOWHEIGHT - 50),    # y coord
            'xs': 0,    # x speed
            'ys': 0,   # y speed
            'w': 20,    # width
            'h': 20,    # height
            'co': GREEN,  # color
            'po': 0,   # points value
            't': 30     # time on screen
            }
        items.append(newItem)

    

    # handle ALL items
    for i in items[:]:
        i['x'] += i['xs']
        i['y'] += i['ys']
        if outside(i, DIMS):
            items.remove(i)
            continue
        if i['t']:
            i['t'] -= 1
            if i['t'] <= 0:
                items.remove(i)

    for i in items:
        cv2.rectangle(frame, (i['x'],i['y']),(i['x']+i['w'],i['y']+i['h']),i['co'],2)

    # handle GUI
    cv2.putText(frame,"Points: %d" %(points),(150,55),cv2.FONT_HERSHEY_COMPLEX,2,255)
    cv2.rectangle(frame, (20, 200),(40, 200-30*health),RED,-1)  # -1 is filled

    if health <= 0:     # game over
        cv2.putText(frame, "GG", (XHALF-100,YHALF),cv2.FONT_HERSHEY_COMPLEX,5,255)
        cv2.imshow('game_window', frame)
        break
    
    # update frame
    cv2.imshow('game_window', frame)

    # check for quitting
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break

##-----------------------------------------------------------------------

cv2.waitKey(0)

##  mission critical
cap.release()
cv2.destroyAllWindows()
