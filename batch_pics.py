import cv2
import time

capLeft = cv2.VideoCapture(0,cv2.CAP_DSHOW)
capLeft.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capLeft.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
capRight = cv2.VideoCapture(1,cv2.CAP_DSHOW)
capRight.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capRight.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
count = 0
loop_size=120
state=True



def picture_loop(camLeft,camRight,loop_size):
    try:
        print(1);time.sleep(1);print(2);time.sleep(1);print(3);time.sleep(2);print('já!')
        pathsLeft=[]
        pathsRight=[]
        framesLeft=[]
        framesRight=[]
        for i in range(loop_size):
            time.sleep(0.5)
            timestamp = time.time_ns()
            retLeft, frameLeft = camLeft.read()
            retRight, frameRight = camRight.read()
            framesLeft.append(frameLeft)
            framesRight.append(frameRight)
            pathsLeft.append(f'left_batch/left_{timestamp}.png')
            pathsRight.append(f'right_batch/right_{timestamp}.png')
        for i in range(loop_size):
            cv2.imwrite(pathsLeft[i], framesLeft[i])
            cv2.imwrite(pathsRight[i], framesRight[i])

    #cv2.imwrite(path, frame)
        return True
    except Exception as e:
        print(e)
        return False
while(count<120):
    if state:
        state = picture_loop(capLeft,capRight,loop_size)
    else:
        print('Failure, Restarting...')
        time.sleep(20)
        state = picture_loop(capLeft,capRight,loop_size)
    count= count + loop_size
    
if state:
    print('Success!')
