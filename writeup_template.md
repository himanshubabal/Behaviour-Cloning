# Behavioral Cloning

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/nvidia_model.png "Model Visualization"
[image2]: ./writeup_images/c_l_r.png "Center Left and Right Camera Images"
[image3]: ./writeup_images/flip.png "Flipped Image"
[image4]: ./writeup_images/trans.png "Translated Image"
[image5]: ./writeup_images/bright.png "Brightness Altered image"
[image6]: ./writeup_images/crop.png "Cropped Image"
[image7]: ./writeup_images/c_r.png "Cropped and resized"
[image8]: ./writeup_images/elu.png "ELU"
[image9]: ./writeup_images/c_r_cl.png "Cropped, resized and RGB2YUV"

## Rubric Points
###### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used NVIDIA Self driving Car model architecture (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

My model consists of a convolution neural network with 3x3 and 5x5 kernel sizes and depths between 3 and 64 (model.py lines 114-126)

The model includes eLU (Exponential Linear Unit) layers to introduce nonlinearity (code line 116-120), Paper link - https://arxiv.org/pdf/1511.07289.pdf

and the data is normalized in the model using a Keras lambda layer (code line 115).

![alt text][image8]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 121).
Dropout Value is kept relatively high (0.50).

 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 131).

#### 4. Appropriate training data

I have used training data provided by udacity.
(https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with a basic fully connected neural network just to test whether the pipeline is working correctly.

Once it was established that my pipeline is working correctly, I started building complex models with convolutional layers in the neural network.
I tried different layer sizes and dropouts, but my model was not behaving the way i wanted it to behave.

Finally, I headed to SDC Slack channel and found out that many students have suggested NVIDIA model, so I gave it a try, and it worked like a charm. It was better then any other model I have tried, so I decided to stick on it.

I used sklearn's test - train split to split data in ratio 20 : 80.

Initially, I trained the model for about 50 epochs, but I found that nearly after 10 epochs, training error and validation error started to diverge, that meant I was overfitting my data, thus I decided to keep my EPOCHS to 10.

Also, I saved only models which show a lower validation error using keras Checkpoint method.

After doing the preprocessing of the data (described below), I fired up the simulator in autonomous mode to see how well my trained model was doing.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.
(video.mp4)

#### 2. Final Model Architecture

The final model architecture (model.py lines 114-126) consisted of a convolution neural network with the following layers and layer sizes


| Output | Shape | # of Param |
|--------|:------|:------|
|lambda_1 (Lambda) | (None, 66, 200, 3)| 0 |
|conv2d_1 (Conv2D) | (None, 31, 98, 24) |1824|
|conv2d_2 (Conv2D) | (None, 14, 47, 36) |21636|
|conv2d_3 (Conv2D) | (None, 5, 22, 48) |43248|
|conv2d_4 (Conv2D) | (None, 3, 20, 64) |27712|
|conv2d_5 (Conv2D) | (None, 1, 18, 64) |36928|
|dropout_1 (Dropout) | (None, 1, 18, 64) |0|
|flatten_1 (Flatten) | (None, 1152) |0|
|dense_1 (Dense) | (None, 100) |115300|
|dense_2 (Dense) | (None, 50) |5050|      
|dense_3 (Dense) | (None, 10) |510|       
|dense_4 (Dense) | (None, 1) | 11 |
|Total params: 252,219|
|Trainable params: 252,219|
|Non-trainable params: 0|


Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

The training dataset obtained from Udacity contained following images in each frame.
Cameras are set up in center and on left and right side of the simulated car.

![alt text][image2]


Processing the data involved augmentation of images.

* Flip the image
    * The track in simulator trains the neural net to take more left turns then right turns. This method takes care of that.

![alt text][image3]

* Translate the image
    * Translating the image in range_x = 100 and range_y = 10. randomly choose the translation amount and translate in x and y. It will be helpful in training the neural net to drive better.

![alt text][image4]

* Alter image brightness
    * It masks the image with shadows to train the neural net to driven correctly even if some part of track is covered in shadows.

![alt text][image5]

* ##### Processing the final image
    * Crop
        * Image was cropped to remove distractions such as car hood and trees. It resulted into better predictions and lower error by neural net.

        ![alt text][image6]

    * Crop and resize
        * Resized image to (66, 200). It resulted into image of similar aspect ratio as of the original image

        ![alt text][image7]

    * Crop, resize and RGB2YUV
        * This transformation was based on NVIDIA architecture as it gives better results.

        ![alt text][image9]

After the collection process, I had 8036 image-frames and since I had 3 images for each frame, I had total of 24,108 training data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by above mentioned experiment.
I used an Adam optimizer so that manually training the learning rate wasn't necessary.
