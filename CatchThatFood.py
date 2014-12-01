## CatchThatFood!
##
## Created by Ethan Robison & Michael Wang
##
## EECS 332 Digital Image Analysis Fall 2014
## Professor Ying Wu
##
import cv2
import pygame, random

# function to check if item is outside the bounds of the screen
def outside(item, screenDim):
    return item.right>screenDim[0] or item.bottom>screenDim[1]

def myDraw(re, im):
    for c in range(0, 3):
        frame[re.top:re.top+im.shape[0],re.left:re.left+im.shape[1],c]=im[:,:,c]*(im[:,:,c]/255.0)+frame[re.top:re.top+im.shape[0],re.left:re.left+im.shape[1],c]*(1.0-oneup[:,:,3]/255.0)


# OpenCV face detection algorithms
faceCascadePath = 'haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(faceCascadePath)
mouthCascadePath = 'haarcascades/Mouth.xml'
mouthCascade = cv2.CascadeClassifier(mouthCascadePath)

# OpenCV webcam feed - loop until frame is valid
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

while frame is None:
    ret, frame = cap.read()

pygame.init()

## PYGAME CONSTANT
WINDOWHEIGHT, WINDOWWIDTH, _ = frame.shape
DIMS = [WINDOWWIDTH, WINDOWHEIGHT]

XHALF = WINDOWWIDTH / 2
YHALF = WINDOWHEIGHT / 2

PBUFF   = 80    # minimun distance from border
TBUFF   = 80
HBUFF   = 100

ITEMSIZE = 30
LIFESIZE = 30

RED     = (  0,   0, 255)
GREEN   = (  0, 255,   0)
BLUE    = (255,   0,   0)

# variables
points = 0  # score
lives = 5

items = []

pointC      = 0     # good item counter
NEWPOINT    = 20    # interval

trapC       = -60   # trap item counter
NEWTRAP     = 30    # interval

healC       = -100  # heal item counter
NEWHEAL     = 50    # interval

mouthRect = pygame.Rect(0,0,0,0)

# pygame sound clips
coin = pygame.mixer.Sound("sound/coin.wav")
power_up = pygame.mixer.Sound("sound/power_up.wav")
power_down = pygame.mixer.Sound("sound/power_down.wav")
hank = pygame.mixer.Sound("sound/hank.wav")
mario_die = pygame.mixer.Sound("sound/mariodie.wav")

# pygame background music
pygame.mixer.music.load("sound/supermario.mp3")
pygame.mixer.music.play(-1)

# pygame images
fruit = [cv2.imread('img/apple.png', -1), cv2.imread('img/peach.png', -1), cv2.imread('img/strawberry.png', -1)]
fruit = [cv2.resize(p, (ITEMSIZE, ITEMSIZE)) for p in fruit]

enemies = [cv2.imread('img/blinky.png', -1), cv2.imread('img/clyde.png', -1), cv2.imread('img/inky.png', -1), cv2.imread('img/pinky.png', -1)]
enemies = [cv2.resize(p, (ITEMSIZE, ITEMSIZE)) for p in enemies]

life = cv2.imread('img/pacman.png', -1)
life = cv2.resize(life, (LIFESIZE, LIFESIZE))

lifeRect = pygame.Rect(10, 150, LIFESIZE, LIFESIZE)

oneup = cv2.imread('img/1up.png', -1)
oneup = cv2.resize(oneup, (ITEMSIZE, ITEMSIZE))

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

            # mouthRect is mouth collision area - BUFFER shrinks it, x, y, and h/2 correct for the fact that lower_mouth coordinates are in the frame of the face detected area
            BUFFER = 10
            mouthRect.x = lowest_mouth[0] + BUFFER + x
            mouthRect.y = lowest_mouth[1] + BUFFER + y + h/2
            mouthRect.w = lowest_mouth[2] - (BUFFER * 2)
            mouthRect.h = lowest_mouth[3] - (BUFFER * 2)

            # cv2.rectangle(roi_color,mouthRect.topleft,mouthRect.bottomright,(0,0,255),2)

            rectList = []
            for i in items:
            	rectList.append(i['rect'])

            isColliding = mouthRect.collidelist(rectList)
            if isColliding > -1:
            	if items[isColliding]['id'] == 0:
            		coin.play()
            	elif items[isColliding]['id'] == 1:
            		power_down.play()
            	elif items[isColliding]['id'] == 2:
            		power_up.play()

            	points += items[isColliding]['po']
            	lives += items[isColliding]['hp']
            	if lives > 5:
            		lives = 5
            	del items[isColliding]
                    
    
    # add items, etc.
    pointC	+= 1
    trapC   += 1
    healC   += 1

    if pointC > NEWPOINT:
        pointC = 1
        newItem = {
        	'id': 0,
            'rect': pygame.Rect(random.randint(PBUFF, WINDOWWIDTH-PBUFF),10,ITEMSIZE,ITEMSIZE),
            'im': fruit[random.randint(0,2)],
            'xs': 0,    # x speed
            'ys': 10,   # y speed
            'po': 10,   # points value
            'hp': 0
            }
        items.append(newItem)
        
    if trapC > NEWTRAP:
        trapC = 1
        newItem = {
        	'id': 1,
            'rect': pygame.Rect(10,random.randint(TBUFF, WINDOWHEIGHT-TBUFF),ITEMSIZE,ITEMSIZE),
            'im': enemies[random.randint(0,3)],
            'xs': random.randint(5, 10), # x speed
            'ys': 0,    # y speed
            'po': 0,    # points value
            'hp': -1     # healing
            }
        items.append(newItem)
        
    if healC > NEWHEAL:
        healC = 1
        newItem = {
        	'id': 2,
            'rect': pygame.Rect(random.randint(HBUFF,WINDOWWIDTH-HBUFF),random.randint(HBUFF,WINDOWHEIGHT-HBUFF),ITEMSIZE,ITEMSIZE),
            'im': oneup,
            'xs': 0,    # x speed
            'ys': 0,    # y speed
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
        if i['id'] == 2:
            i['t'] -= 1
            if i['t'] <= 0:
                items.remove(i)

    for i in items:
        myDraw(i['rect'], i['im'])

    # handle GUI
    cv2.putText(frame,"Points: %d" %(points),(150,55),cv2.FONT_HERSHEY_COMPLEX,2,255)
    for hp in range(lives):
        lifeRect.y = (200 - hp*(LIFESIZE+10))
        myDraw(lifeRect, life)

    if lives <= 0:     # game over
        cv2.putText(frame, "GG", (XHALF-100,YHALF),cv2.FONT_HERSHEY_COMPLEX,5,255)
        cv2.imshow('game_window', frame)

        pygame.mixer.music.stop()
        mario_die.play()
        hank.play()

        cv2.waitKey(0)
        break
    
    # update frame
    cv2.imshow('game_window', frame)

    # check for quitting
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break



##  mission critical
cap.release()
cv2.destroyAllWindows()