import cv2
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def calibrateCamera():
    objpoints = []
    imgpoints = []
    objp = np.zeros((9*6,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    dir = 'camera_cal/'
    files = listdir(dir)
    for file in files:
        img = mpimg.imread(dir+file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx,dist

def test_calibrate():
    mtx,dist = calibrateCamera()
    dir = 'camera_cal/'
    img = mpimg.imread(dir+'calibration1.jpg')
    dst = cv2.undistort(img, mtx, dist)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(dst)
    plt.show()
    
if __name__ == '__main__':
    calibrateCamera()
    test_calibrate()
    print('end')