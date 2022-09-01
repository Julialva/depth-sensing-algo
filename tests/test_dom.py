import numpy as np
import cv2
import matplotlib.pyplot as plt

imgL = cv2.imread('C:/Users/gabri/OneDrive/Desktop/depth-sensing-algo-main/zip_pics/left/img_1660695748.png',0)
imgR = cv2.imread('C:/Users/gabri/OneDrive/Desktop/depth-sensing-algo-main/zip_pics/right/img_1660695748.png',0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=7)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()