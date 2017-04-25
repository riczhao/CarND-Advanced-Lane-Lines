##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---
[//]: # (Image References)

[image0]: ./examples/overview.png "overview"
[image1]: ./examples/my_undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/rbgGradient.png "rgb gradient Example"
[image4]: ./examples/my_warp.png "Warp Example"
[image5]: ./examples/poly.png "Fit Visual"
[image6]: ./examples/marked.png "Output"
[video1]: ./mark_project_video.mp4 "Video"

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color gradients, etc., to create a gradient image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

I tried HLS, but it looks S or L changes a lot at far end of lane line. So I moved to
caculate gradient of RGB color. I also tryied creating binary image to detect lane lines, but it dropped gradient value, which caused a lot of noise in histogram. Using
gradient image directly is better.
![alt text][image0]

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is function calibrateCamera (file detect.py) 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I created a class Image to include all transformation for a single image.

The code for my perspective transform includes a function called `Image.perspectiveMatrix()`.  I chose the hardcode the source and destination points in the following manner:

```
       src = np.array([
            [601, 446], [683, 446],
            [266, 671], [1037, 671],
            ]).astype(np.float32)
       dst = np.array([
            [411, 20], [885, 20],
            [411, 1420], [885, 1420],
            ]).astype(np.float32)
 
```
I verified that my perspective transform was working as expected by verifying that the lines appear parallel in the warped image.

![alt text][image4]

####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

As I stated earlier, I didn't use binary image but use gradient image directly. I calculate X gradient for R,G,B channels seperately and get norm of the three value.

![alt text][image3]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I sum bottom half of the rbg gradient image on y direction, to get a histogram. And detect lane line start points by finding two points that's closest to middle and pass threshhold.
Then I use a 50x50 slide window to move up and find all lane points.

And fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in funciton `Image.curverad()`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `Image.markImg()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

