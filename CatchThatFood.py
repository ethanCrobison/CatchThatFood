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
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

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

    # update frame
    cv2.imshow('game_window', frame)

    # check for quitting
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break

##-----------------------------------------------------------------------


##  mission critical
cap.release()
cv2.destroyAllWindows()
