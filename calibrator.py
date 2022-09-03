import numpy as np
import cv2
import glob
import os
from picture_api.utils.image_utils import show_image

def calibrate(frame_size:tuple=(640,360),chessboard_size:tuple=(8,6),image_dir:str=''):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    #os.chdir(dir)

    images = glob.glob(image_dir)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #show_image(img)
        #show_image(gray)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        #cv2.imshow('img',gray)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            #cv2.drawChessboardCorners(img, chessboard_size, corners,ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)
        else:
            raise "ImageError: Chessboard not found."
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return ret, mtx, dist, rvecs, tvecs
    #cv2.destroyAllWindows()



if __name__ =="__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate(image_dir='calib/right_cal/*.png')
    img = cv2.imread('calib/right_cal/right_1661541497272835700.png')
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    show_image(dst)
    cv2.imshow('img',img)
    cv2.imwrite('calibresult.png',dst)