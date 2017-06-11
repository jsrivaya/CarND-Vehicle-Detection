
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG_car_nocar.png
[image2]: ./output_images/YUV_test_images.png
[image3]: ./output_images/YUV_windows_heatmap.png
[image4]: ./output_images/YUV_threshold.png
[image5]: ./output_images/YUV_processed.png
[image6]: ./examples/YUV_processed.png
[video1]: ./project_video_processed.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the second code IPython notebook. I use the first for imports.

  ```python
  # Define a function to return HOG features and visualization
  def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                          vis=False, feature_vec=True):
      # Call with two outputs if vis==True
      if vis == True:
          features, hog_image = hog(img, orientations=orient, 
                                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                                    cells_per_block=(cell_per_block, cell_per_block), 
                                    transform_sqrt=False, 
                                    visualise=vis, feature_vector=feature_vec)
          return features, hog_image
      # Otherwise call with one output
      else:      
          features = hog(img, orientations=orient, 
                         pixels_per_cell=(pix_per_cell, pix_per_cell),
                         cells_per_block=(cell_per_block, cell_per_block), 
                         transform_sqrt=False, 
                         visualise=vis, feature_vector=feature_vec)
          return features
  ```
  
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes with it's associated HOG image.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB`, `HLS`, `YCrCb`, and `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

I tried various combinations of parameters. I first started with `HLS` because it would through would results in the training but looking into the HOG images dind't look very promising. I then moved to `YCrCb` and did most of the work in this colorspace. I realized in the image processing that this colosspace had a really hard time classifing bright colors like white. I then moved to 'YUV' and found that suddenly it was more easily recognizing white car. For the orientation I followed the vehicle recognition paper suggested where it says that an orientation of 9 for the HOG through the best results.

I trained a linear SVM as suggested in the lesson. I do this in the next piece of code, located also in the 6 code cell of the IPython Jupyter book.

```python
# Train Model.
t=time.time()

n_samples = 1228
random_idx = np.random.randint(0, len(cars), n_samples)
test_cars = np.array(cars)[random_idx]
random_idx = np.random.randint(0, len(notcars), n_samples)
test_notcars = np.array(notcars)[random_idx]

car_features = extract_features(test_cars, color_space=color_space, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(test_notcars, color_space=color_space, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC ########################################
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
# #########################################################

```

Is good to point out that I augmented the dataset (up to 1228 images), to keep a well balanced set of images between cars and not cars. I also randomly initialized the training set of images:

```python
n_samples = 1228
random_idx = np.random.randint(0, len(cars), n_samples)
test_cars = np.array(cars)[random_idx]
random_idx = np.random.randint(0, len(notcars), n_samples)
test_notcars = np.array(notcars)[random_idx]

car_features = extract_features(test_cars, color_space=color_space, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(test_notcars, color_space=color_space, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
```

### Sliding Window Search

I tried different approaches for windows searching. I did start with a window size of 64 because this was the training size. I also started with an overlap of .5 because this was the suggested in class. In one part of the class was suggested to try different window sizes depending on which part of the image we were looking for cars. So my initial approach was to change window size depending on which section of the image I would look for. Then I append all the features extration from each section. From `ystart=400` to `ystop=500` small windows of `64`, from `ystart=500` to `ystop=600` medium windo of `96` and from `ystart=550` to `ystop=650` big window of `128`. This ended up to be too messy and hard to debug so I moved to a fix window size that would through the best results in general and decided to work on the false positives via the threshold and heatmap buffering.
Bellow is the pice of code that compute the sliding window search

```python
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list
```

In the following image I show the testing images using the final colorspace `YUV`

![alt text][image2]

Ultimately I searched on a 1.65 scale using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I do compute aprox 850 windows per frame. I also buffer every heatmap for the last 5 frames to reduce the number of false positive keeping them to a minimum.

Here are some example images of the window and heatmap:

![alt text][image3]

And here the final frame result:

![alt text][image4]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

