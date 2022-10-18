
import numpy as np
import cv2
q = [*np.ones(40)]
y = [*range(40)]
img = np.zeros((512, 512, 3), np.int32)
curve = np.column_stack((q, y))
x = cv2.polylines(img, [curve.astype(np.int32)], False, (255, 0, 0))

cv2.imshow("sss", x.astype(np.uint8))
