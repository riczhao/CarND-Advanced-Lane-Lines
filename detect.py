import cv2
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sqlalchemy.sql.expression import false

idx = 1

'''
pipeline:
- calibrate camera
- get s 
'''
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

class Image(object):
    def __init__(self, fn=None,img=None):
        if fn:
            self.filename = fn
            self.img = mpimg.imread(self.filename)
        else:
            self.img = img
        self.top_clip = 0
        self.img = self.img[self.top_clip:,:,:]
        self.useLast = False

    camera_mtx,camera_dist = calibrateCamera()

    def undistort(self):
        self.img_undist = cv2.undistort(self.img, Image.camera_mtx, Image.camera_dist)

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
    @classmethod
    def toHLS(cls, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        return hls[:,:,0],hls[:,:,1],hls[:,:,2]
    @classmethod
    def clamp(cls, img, thresh=(0,255)):
        img[(img >= thresh[0]) & (img <= thresh[1])] = 0
        return img
    def perspectiveMatrix(self):
        src = np.array([
            #[609, 442], [671, 442], #straight 1
            #[270, 673], [1036, 673],
            [601, 446], [683, 446],
            [266, 671], [1037, 671],
            ]).astype(np.float32)
        '''
        dst = np.array([
            [411, 442], [885, 442],
            [411, 579], [885, 579],
            ]).astype(np.float32)
            '''
        dst = np.array([
            [411, 20], [885, 20],
            [411, 1420], [885, 1420],
            ]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        self.RM = cv2.getPerspectiveTransform(dst, src)
        self.M = M
    def checkColor(self, color):
        pass
    @classmethod
    def rgbGrad(cls,img,min=21.814):
        order=7
        imgR = cv2.Sobel(img[:,:,0], cv2.CV_64F, 1, 0, ksize=order)
        imgG = cv2.Sobel(img[:,:,1], cv2.CV_64F, 1, 0, ksize=order)
        imgB = cv2.Sobel(img[:,:,2], cv2.CV_64F, 1, 0, ksize=order)
        norm = (imgR**2+imgG**2+imgB**2)**0.5
        normX = norm*255./norm.max()
        '''
        imgR = cv2.Sobel(img[:,:,0], cv2.CV_64F, 0, 1, ksize=order)
        imgG = cv2.Sobel(img[:,:,1], cv2.CV_64F, 0, 1, ksize=order)
        imgB = cv2.Sobel(img[:,:,2], cv2.CV_64F, 0, 1, ksize=order)
        norm = (imgR**2+imgG**2+imgB**2)**0.5
        normY = norm*255./norm.max()
        plt.imshow(normY,cmap='gray')
        '''
        mask = np.zeros_like(normX)
        mask[normX>=min] = 1.
        return mask,normX
    def detect(self):
        h,l,s = Image.toHLS(self.img)
        l = l/l.max()
        s = s/s.max()
        img = self.img
        per = cv2.warpPerspective(self.img, self.perspective(), (1280,720*2))
        s = cv2.warpPerspective(s, self.perspective(), (1280,720*2))
        bin,_ = Image.rgbGrad(per, min=21.814) #21.814
        histogram = np.sum(bin[bin.shape[0]//2:], axis=0)
        return bin,histogram
        '''
        plt.subplot(2,3,1)
        plt.imshow(bin, cmap="gray")
        plt.subplot(2,3,2)
        plt.imshow(per)
        plt.subplot(2,3,3)
        plt.imshow(s,cmap='gray')
        plt.subplot(2,3,4)
        plt.plot(histogram)
        plt.show()
        '''
        '''
        grad = cv2.Sobel(s, cv2.CV_64F, 1, 0, ksize=3)
        grad = np.absolute(grad)
        binary = np.zeros_like(s)
        binary[(s>20)&(s<25)] = 1
        plt.subplot(1,2,1)
        plt.imshow(s,cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(grad,cmap='gray')
        plt.show()
        '''
    def detectLaneStart(self):
        hist_th = 6300
        hit = False
        left_regions = [] # (Xstart, Xend, Xmax)
        right_regions = []
        region = [0,0,0]
        hist = self.grad_hist
        # search left lane line
        for x in range(hist.shape[0]//2,-1,-1):
            if hist[x] > hist_th:
                if not hit:
                    hit = True
                    region[0] = x
                    continue
                else:
                    continue
            else:
                if hit:
                    region[1] = x
                    max_x = np.argmax(hist[region[1]-1:region[0]+1]) + region[1]
                    region[2] = max_x
                    left_regions.append(region)
                    region = [0,0,0]
                    hit = False
                else:
                    continue
        # search right lane line
        for x in range(hist.shape[0]//2, hist.shape[0]):
            if hist[x] > hist_th:
                if not hit:
                    hit = True
                    region[0] = x
                    continue
                else:
                    continue
            else:
                if hit:
                    region[1] = x
                    max_x = np.argmax(hist[region[0]:region[1]]) + region[0]
                    region[2] = max_x
                    right_regions.append(region)
                    region = [0,0,0]
                    hit = False
                else:
                    continue
        self.left_lane_start = np.argmax(hist[300:640]) + 300
        self.right_lane_start = np.argmax(hist[640:980]) + 640
        #self.left_lane_start = left_regions[0][2]
        #self.right_lane_start = right_regions[0][2]
        if self.right_lane_start-self.left_lane_start < 100:
            print('hit')
            self.useLast = True
 
    def curverad(self, fit, y):
        return ((1 + (2*fit[0]*y + fit[1])**2)**1.5) / np.absolute(2*fit[0]) * 0.0077244

    def detectLanes(self):
        global idx
        self.flat_shape = (1280,720*2)
        self.undistort()
        self.perspectiveMatrix()
        self.img_flat = cv2.warpPerspective(self.img, self.M, self.flat_shape)
        self.img_bin,self.img_rgbGrad = Image.rgbGrad(self.img_flat)
        self.img_bin = self.img_rgbGrad
        self.grad_hist = np.sum(self.img_bin[self.img_bin.shape[0]//2:], axis=0)
        self.detectLaneStart()
        self.left_points = self.slideWindow(self.left_lane_start)
        self.right_points = self.slideWindow(self.right_lane_start)
        left_fit,left_points = self.polynomial(self.left_points)
        right_fit,right_points = self.polynomial(self.right_points)
        if np.absolute((right_points[-1][0] - left_points[-1][0]) - (right_points[0][0] - left_points[0][0])) > 50:
            print('w hit')
            self.useLast = True
        self.left_fit = left_fit
        self.right_fit = right_fit
        self.curve = self.curverad(right_fit, 1200) * 10
        self.offset = ((right_points[-1][0]+left_points[-1][0])/2 -  740) * 0.0077244
        self.img_marked = self.markImg(left_points, right_points)
        self.left_poly_points = left_points
        self.right_poly_points = right_points
        return

        '''
        sliding windows
        '''
        left_start = left_regions[0][2]
        right_start = right_regions[0][2]
        left_points = self.slideWindow(bin, left_start)
        right_points = self.slideWindow(bin, right_start)
        ''' draw lane points '''
        bin = np.dstack([bin,bin,bin])
        for pt in left_points:
            cv2.circle(bin, pt, 10, (1.,0,0), 5)
        for pt in right_points:
            cv2.circle(bin, pt, 10, (0,1.0,0), 5)
        left_fit,left_points = self.polynomial(bin, left_points)
        right_fit,right_points = self.polynomial(bin, right_points)
        self.img = self.markImg(left_points, right_points)
        ''' polynomial
        plt.subplot(2,6,idx)
        idx += 1
        plt.imshow(self.img)
        plt.subplot(2,6,idx)
        idx += 1
        plt.plot(hist)
        print(right_regions[0][0]-left_regions[0][0], left_regions[0][0], right_regions[0][0])
        '''
    def slideWindow(self, start):
        '''
        return points of lane line
        '''
        bin = self.img_bin
        win_w = 50
        win_h = 50
        points = []
        points.append((start, bin.shape[0]-1))
        win_mid = start
        win_thr = 800
        for y in range(bin.shape[0]-1,win_h,-win_h):
            win = bin[y-win_h:y, win_mid-win_w:win_mid+win_w]
            sum = np.sum(win,axis=0)
            if len(sum)==0:
                continue
            x = np.argmax(sum)
            if sum[x] < win_thr:
                continue
            ofs = x-50 if x>=50 else 0
            smooth = sum[ofs:x+50]
            idx = np.arange(len(smooth))
            x = int(np.sum(smooth*idx)/np.sum(smooth)) + ofs
            win_mid = x + win_mid - win_w
            points.append((win_mid, y - win_h//2))
        return points
    def polynomial(self, points):
        bin = self.img_bin
        points = np.array(points)
        fit = np.polyfit(points[:,1],points[:,0],2)
        ploty = np.linspace(0, bin.shape[0]-1, bin.shape[0])
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        points = np.int32([np.vstack([fitx,ploty]).T])
        #cv2.polylines(bin, points, False, (1.,0,0), 10)
        return fit,points[0]
    left_points = None
    def markImg(self, left_points, right_points):
        if self.useLast and Image.left_points is not None:
            left_points = Image.left_points
            right_points = Image.right_points
        if not self.useLast:
            Image.left_points = left_points
            Image.right_points = right_points
        mask = np.zeros((720*2,1280,3),dtype=np.uint8)
        polygon = np.concatenate((left_points,right_points[::-1]))
        cv2.fillPoly(mask, np.int_([polygon]), (0,255,0))
        #plt.imshow(mask)
        overlay = cv2.warpPerspective(mask, self.RM, (self.img.shape[1],self.img.shape[0]))
        img = cv2.addWeighted(self.img,1,overlay,0.3,0)
        cv2.putText(img, 'Radius of Curvature = {}(m)'.format(int(self.curve)),
                    (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0))
        if self.offset < 0:
            cv2.putText(img, 'Vehicle is {:10.4f}(m) left of center'.format(np.absolute(self.offset)),
                    (30,90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0))
        else:
            cv2.putText(img, 'Vehicle is {:10.4f}(m) right of center'.format(np.absolute(self.offset)),
                    (30,90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0))
        return img
    def showImages(self):
        self.detectLanes()
        bin = np.dstack([self.img_bin,self.img_bin,self.img_bin])
        bin = bin/bin.max()
        for pt in self.left_points:
            cv2.circle(bin, pt, 10, (1.,0,0), 5)
        for pt in self.right_points:
            cv2.circle(bin, pt, 10, (0,1.0,0), 5)
        cv2.polylines(bin, np.int32([self.left_poly_points]), False, (1.,0,0), 5)
        cv2.polylines(bin, np.int32([self.right_poly_points]), False, (0,1.,0), 5)
        hist = np.sum(self.img_rgbGrad[self.img_rgbGrad.shape[0]//2:], axis=0)

        plt.subplot(2,3,1)
        plt.imshow(self.img)
        plt.subplot(2,3,2)
        plt.imshow(self.img_flat)
        plt.subplot(2,3,3)
        plt.imshow(self.img_rgbGrad,cmap='gray')
        plt.subplot(2,3,4)
        plt.imshow(bin)
        plt.subplot(2,3,5)
        plt.plot(self.grad_hist)
        plt.subplot(2,3,6)
        plt.imshow(self.img_marked)
        plt.show()

from moviepy.editor import VideoFileClip
def processImage(img):
    image = Image(img=img)
    image.detectLanes()
    return image.img_marked
def markVideo(fn):
    white_output = 'mark_'+fn
    clip1 = VideoFileClip(fn)
    white_clip = clip1.fl_image(processImage)
    white_clip.write_videofile(white_output, audio=False)

if __name__ == '__main__':
    test = False
    if test:
        Image('test_images/test1.jpg').showImages()
        img = Image('test_images/straight_lines2.jpg').showImages()
        for i in range(1,10):
            Image('t'+str(i)+'.jpg').showImages()
        Image('t1.jpg').showImages()
        Image('2.jpg').showImages()
        Image('3.jpg').showImages()
        Image('4.jpg').showImages()
        Image('5.jpg').showImages()
        Image('test_images/test1.jpg').showImages()
        Image('test_images/test2.jpg').showImages()
        Image('test_images/test3.jpg').showImages()
        Image('test_images/test4.jpg').showImages()
        Image('test_images/test5.jpg').showImages()
        Image('test_images/test6.jpg').showImages()
    else:
        markVideo('project_video.mp4')
    print('end')