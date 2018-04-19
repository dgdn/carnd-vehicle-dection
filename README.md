**Vehicle Detection Project**

Here is the writeup for SDC-P5.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector.
* Normalize features and randomize a selection for training and testing.
* Using sliding-window technique and trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Extract HOG features from the training images

The code for this step is contained in lines 45 through 57 of the file called `classification.py`).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Settled on final choice of HOG parameters

I tried various combinations of parameters and here was the final one that gave best performance. `color_space=YCrCb`, `orientations=9`, `cell_per_block=2`, `pixels_per_cell=(8,8)`, and use 3 channels.

#### 3. Trained a classifier using selected HOG features and color features

I trained a linear SVM using `sklearn.svm.LinearSVC` with default parameters. The train datas contains 17760 samples and the features length was 8460. The Linear svc gave accuray 98.99% on test set. I had aslo experimented the svm with rbf kernel, it take extremely long time to train and reached accuracy 99.45%. Though it had higher accuracy I didn't use it for detection because it also take extramely long time for predict.

### Sliding Window Search

The code for this step is contained in lines 76 through 105 of the file called `detection.py`.

I slided the widow on image from left to right and from up to bottom. Empircally, the step size may be 1/4 of window size thus each two adjucent windows may have 3/4 overlap area. For each patch of window, svm classication was runned on it to judge whether a car was located in it. If we had high confidence that there was a car in it which has high svm prediction score, we will mark this area with a bounding box indicate that a car located in it. After one pass through sliding window search, we end up with multiple bounding boxes, some of which may overlap. Finally, we merge the overlap bounding box into bigger one. The resulted bounding boxes were the location of cars that we want to find. That was the overall process.

To improve performance of the process, I did not slide over the entire images because the cars only appear in half bottom part of the image. Also as known that we extracted HOG features for each window, we can merge this operations into extracting the whole area of HOG features once and slide windows on HOG feature map instead of original image. Accordingly, we had to tranform the window size and step in pixel metrics into HOG features metrics.

To better detect difference sizes of cars in the images, we need to slide window on more than one scale of the images. The scale was bigger than 1. Size 1 is the original image size, the bigger scale indicate that the slide window may cover bigger area. Thus small scale is better at caputure small cars and big scale is better at caputure the big one.

I decided to search window positions at two scales(1, 1.5) all over the image and came up with this

![alt text][image3]

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

---

### Video Implementation

Here's a [link to my video result](./project_video_output.mp4)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image6]

 Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]

---

### Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

At the begining of training classifier, although I had exaughted the combination of extract feature params, the accuracy of classifier could not reach above 99%. Finally, I found that the mistake come from I set the range of the color histogram bin size to be always between 1 and 255. After removed this restriction the accuracy reached 99%.

I had spended tons of time tuning the detection parameters including `heat_threshold`, `score_threshold`, and `scale`. I found it was really hard for me to make it work well. The biggest problem was that the svm classicfication can not generalize well to video frames. Sometime it may detect so many false positive such that even though I apply the heatmap and threshold predict socre it will not work well. Increased the predict score can reduce the false positives, but the cars also got lost. Maybe I should spend more time to improve the svm classifier.

Another problem may be low performance of pipline. The pipeline need to take 1 second to process 1 image! That was not practical for realtime detection.