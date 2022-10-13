import cv2

cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


while (True):
    ret, frame = cap1.read()
    ret2, frame2 = cap2.read()
    cv2.imshow('OpenCv', frame)
    cv2.imshow('OpenCv2', frame2)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
