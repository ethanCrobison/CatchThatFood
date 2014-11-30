##
## CatchThatFood!
##
## Created by Ethan Robison & Michael Wang
##
## EECS 332 Digital Image Analysis Fall 2014
## Professor Ying Wu
##
import cv2
import pygame, random


##--------------- Functions ---------------------------------------------
def within(outer, inner):
    print 'Placeholder'

def outside(item, screenDim):
    return item.right>screenDim[0] or item.bottom>screenDim[1]
        


##--------------- Initialize --------------------------------------------

# pygame
pygame.init()

# music
pygame.mixer.music.load("mp3/supermario.mp3")

# openCV
cap = cv2.VideoCapture(0)   # instantiate webcam feed

ret, frame = cap.read()     # check that frame is valid
while frame is None:
    ret, frame = cap.read()

faceCascadePath = 'haarcascades/haarcascade_frontalface_default.xml'   # instantiate face detection
faceCascade = cv2.CascadeClassifier(faceCascadePath)
mouthCascadePath = 'haarcascades/Mouth.xml'
mouthCascade = cv2.CascadeClassifier(mouthCascadePath)

##--------------- Game Stuff --------------------------------------------
# constants
WINDOWHEIGHT, WINDOWWIDTH, depth = frame.shape  # image dimensions
DIMS = [WINDOWWIDTH, WINDOWHEIGHT]

XHALF = WINDOWWIDTH / 2
YHALF = WINDOWHEIGHT / 2

PBUFF   = 80    # minimun distance from border
TBUFF   = 80
HBUFF   = 100

RED     = (  0,   0, 255)
GREEN   = (  0, 255,   0)
BLUE    = (255,   0,   0)

# variables
points = 0  # score
health = 5  # "health"

items = []

pointC      = 0     # good item counter
NEWPOINT    = 20    # interval

trapC       = -60   # trap item counter
NEWTRAP     = 30    # interval

healC       = -100  # heal item counter
NEWHEAL     = 50    # interval

mouthRect = pygame.Rect(0,0,0,0)

pygame.mixer.music.play(loops=-1)

coin = pygame.mixer.Sound("mp3/coin.wav")
power_up = pygame.mixer.Sound("mp3/power_up.wav")
power_down = pygame.mixer.Sound("mp3/power_down.wav")


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
        roi_gray = gray[y+h/2:y+h, x:x+w]
        roi_color = frame[y+h/2:y+h, x:x+w]

        mouths = mouthCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(50, 50),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        if list(mouths):
            y_values = zip(*mouths)[1]
            max_index = y_values.index(max(y_values))
            lowest_mouth = mouths[max_index]

            BUFFER = 10
            mouthRect.x = lowest_mouth[0]
            mouthRect.x += BUFFER
            mouthRect.y = lowest_mouth[1]
            mouthRect.y += BUFFER
            mouthRect.w = lowest_mouth[2]
            mouthRect.w -= (2*BUFFER)
            mouthRect.h = lowest_mouth[3]
            mouthRect.h -= (2*BUFFER)
            cv2.rectangle(roi_color,mouthRect.topleft,mouthRect.bottomright,(0,0,255),2)

            mouthRect.x += x
            mouthRect.y += y+h/2

            rectList = []
            for i in items:
            	rectList.append(i['rect'])

            isColliding = mouthRect.collidelist(rectList)
            if isColliding > -1:
            	if items[isColliding]['po'] > 0:
            		coin.play()
            	elif items[isColliding]['hp'] > 0:
            		power_up.play()
            	else:
            		power_down.play()

            	points += items[isColliding]['po']
            	health += items[isColliding]['hp']
            	if health > 5:
            		health = 5
            	del items[isColliding]
                    
    
    # add items, etc.
    pointC  += 1
    trapC   += 1
    healC   += 1

    if pointC > NEWPOINT:
        pointC = 1
        newItem = {
            'rect': pygame.Rect(random.randint(PBUFF, WINDOWWIDTH-PBUFF),10,10,10),
            'xs': 0,    # x speed
            'ys': 10,   # y speed
            'co': RED,  # color
            'po': 10,   # points value
            't': 0,
            'hp':0 
            }
        items.append(newItem)
        
    if trapC > NEWTRAP:
        trapC = 1
        newItem = {
            'rect': pygame.Rect(10,random.randint(TBUFF, WINDOWHEIGHT-TBUFF),10,10),
            'xs': random.randint(5, 10), # x speed
            'ys': 0,    # y speed
            'co': BLUE, # color
            'po': 0,    # points value
            't': 0,
            'hp':-1     # healing
            }
        items.append(newItem)
        
    if healC > NEWHEAL:
        healC = 1
        newItem = {
            'rect': pygame.Rect(random.randint(HBUFF,WINDOWWIDTH-HBUFF),random.randint(HBUFF,WINDOWHEIGHT-HBUFF),20,20),
            'xs': 0,    # x speed
            'ys': 0,    # y speed
            'co': GREEN,# color
            'po': 0,    # points value
            't': 30,    # time on screen
            'hp': 1     # healing
            }
        items.append(newItem)

    

    # handle ALL items
    for i in items[:]:
        i['rect'].move_ip(i['xs'],i['ys'])
        if outside(i['rect'], DIMS):
            items.remove(i)
            continue
        if i['t']:
            i['t'] -= 1
            if i['t'] <= 0:
                items.remove(i)

    for i in items:
        cv2.rectangle(frame, i['rect'].topleft,i['rect'].bottomright,i['co'],2)

    # handle GUI
    cv2.putText(frame,"Points: %d" %(points),(150,55),cv2.FONT_HERSHEY_COMPLEX,2,255)
    cv2.rectangle(frame, (20, 200),(40, 200-30*health),RED,-1)  # -1 is filled

    if health <= 0:     # game over
        cv2.putText(frame, "GG", (XHALF-100,YHALF),cv2.FONT_HERSHEY_COMPLEX,5,255)
        cv2.imshow('game_window', frame)
        cv2.waitKey(0)
        break
    
    # update frame
    cv2.imshow('game_window', frame)

    # check for quitting
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break

##-----------------------------------------------------------------------

##  mission critical
cap.release()
cv2.destroyAllWindows()
