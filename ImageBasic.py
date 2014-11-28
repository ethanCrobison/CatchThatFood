import numpy as np
import cv2

img = cv2.imread('Comfy_Doge.jpg', 0)
cv2.imwrite('ComfyDogeGray.jpg', img)
##cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
##cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
