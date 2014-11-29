##
## CatchThatFood!
##
## Created by Ethan Robison & Michael Wang
##
## EECS 332 Digital Image Analysis Fall 2014
## Professor Ying Wu
##
import cv2

# instantiate webcam feed
# pass 0 or -1 to use default cam
cap = cv2.VideoCapture(0)

# check that frame is valid
ret, frame = cap.read()
while frame is None:
    ret, frame = cap.read()

height, width, depth = frame.shape

# instantiate face detection
faceCascadePath = 'haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(faceCascadePath)
mouthCascadePath = 'haarcascades/haarcascade_mcs_mouth.xml'
mouthCascade = cv2.CascadeClassifier(mouthCascadePath)

##--------------- Main Loop ---------------------------------------------
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)	# mirror orientation for game

    # face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
    	gray,
    	scaleFactor=1.1,
    	minNeighbors=5,
    	minSize=(100, 100),
    	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # mouth-in-face detection
    for (x,y,w,h) in faces:
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    	roi_gray = gray[y:y+h, x:x+w]
    	roi_color = frame[y:y+h, x:x+w]

    	mouths = mouthCascade.detectMultiScale(
    		roi_gray,
	    	scaleFactor=1.1,
	    	minNeighbors=50,
	    	minSize=(50, 30),
	    	flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    	)

    	for (mx,my,mw,mh) in mouths:
    		cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)


    # update frame
    cv2.imshow('game_window', frame)

    # check for quitting
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break

##-----------------------------------------------------------------------


##  mission critical
cap.release()
cv2.destroyAllWindows()
