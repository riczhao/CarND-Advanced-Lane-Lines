import cv2
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

'''
pipeline:
- calibrate camera
- get s 
'''

class Image(object):
    def __init__(self, fn):
        self.filename = fn
        self.img = mpimg.imread(self.filename)
    @classmethod
    def calibrateCamera(cls):
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
    @classmethod
    def test_calibrate(cls):
        mtx,dist = calibrateCamera()
        dir = 'camera_cal/'
        img = mpimg.imread(dir+'calibration1.jpg')
        dst = cv2.undistort(img, mtx, dist)
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(dst)
        plt.show()
    def toHLS(self):
        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
        for i in range(3):
            s = hls[:,:,i]
            #s = s/s.max()
            plt.subplot(1,3,i+1)
            plt.imshow(s,cmap='gray')
        plt.show()
    def detectYellow(self):
        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
        s = hls[:,:,2]
        grad = cv2.Sobel(s, cv2.CV_64F, 1, 0, ksize=5)
        grad = np.absolute(grad)
        max = 25
        min = 18
        binary = np.zeros_like(s)
        binary[(s>20)&(s<25)] = 1
        plt.subplot(1,2,1)
        plt.imshow(s,cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(grad,cmap='gray')
        plt.show()

if __name__ == '__main__':
    #img = Image('test_images/straight_lines1.jpg')
    img = Image('test_images/test5.jpg')
    img.detectYellow()
    #img.toHLS()
    print('end')