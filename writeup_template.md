# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of examples in each class in each dataset (train,validation,test)


[image1]: label-counts.png "Label Counts"
This shows a visual representation of the label counts from each data set.  Here we can see that each set has the same proportion of labels and that the speed signs (sub 15 labels) make a majority of the labels.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Overall, I wanted to see how individual small changes affected accuracy.  For preprocessing,  I wanted to see what would happen with only minimal preprocessing (ie shuffling).  Later on I tried normalization.  I tried the grayscaling, but it didn't yield the results I wanted.  I found another normalization method, and it seemed to yield better results.  I'm not exactly sure why or what it is called.  I suspect that it may be because this is a type of mean normalization.

Interestingly enough, I was able to get to 94% without the other suggestions of grayscaling and adding additional data.  

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stried, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| output: 400									|
| Fully connected		| output 120        							|
| RELU					|												|
| Dropout 				| keep rate 0.9									|
| Fully connected		| output 84        								|
| RELU					|												|
| Dropout 				| keep rate 0.9									|
| Fully connected		| output 43        								|
| Softmax				| 43 classes        							|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

As with the LeNet implementation, I left the default adam optimizer, learning_rate of 0.001, batch size of 128, mu of 0, and sigma of 0.1.  For the number of epochs, I found that increasing it significantly increased the accurracy.  I initially tried to leave it at 20 because it was taking longer and longer to train on my laptop.  Eventually I settled on 60.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.942
* test set accuracy of 0.917

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

LeNet - that was the suggested architecture, and the problem seemed similar.  Numbers are similar to signs in that it is a few lines and simple shapes.

* What were some problems with the initial architecture?

It was underfitting and not accurate enough.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Initially, I modified a bunch of parameters, so I could observe the changes in accuracy.  This ranged from everything mu,sigma, learning rate, batch size etc.  I tried increasing the Fully Connected layer node size, but it did not really help enough to get me to 93%.  Eventually, I was getting no where, so I looked to the forums for some guidance.  I added more epochs which resulted in the validation set increasing.  This made sense as the accuracy was not decreasing yet.  However, it did not do well on the test set.  Thus I added two drop out layers to reduce the overfitting, and after testing various parameters, I settled on a 0.9 keep rate.

* Which parameters were tuned? How were they adjusted and why?

I modified the drop out ratio and epochs.  Epochs were increased to increase accuracy.  However, when underfitting, I added two drop out layers.  I modified a bunch of different paramteres, but it didn't seem to help.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convultion layer work well because it uses the knowledge of the image structure.  Drop out layers seemed to help so that it helped the network learn different patterns for each class in addition to the convolution layers.

If a well known architecture was chosen:
* What architecture was chosen?
	LeNet
* Why did you believe it would be relevant to the traffic sign application?
	Similarly to classifying numbers, I figured it would work well with classifying traffic signs.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 Both the validation and test accuracy were high leading me to believe the model is working well.  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web

[image4]: newTestImages/11.jpg "right-of-way"
[image5]: newTestImages/12.jpg "priority road" 
[image6]: newTestImages/17.jpg "no entry" 
[image7]: newTestImages/25.jpg "road work" 
[image8]: newTestImages/4.jpg "70 km/h limit" 
<!-- ![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]
 -->
The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
I downloaded images from the inernet, and then used some free website to resize it rather than python since there was only 5 images.  
It guessed accruacy of 60% For the simpler images, it guessed correctly (right-of-way,priority road, road work).  For the two it guessed incorrectly, there was a lot of noise in the images.
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| right-of-way      	| right-of-way   								| 
| Priority road  		| Priority road 								|
| No Entry				| Priority road									|
| Road Work	      		| Road Work					 					|
| 70 km/h limit			| General Caution      							|




#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Interestingly enough, when I found out the probabilities, I thought maybe they were wrong (ie there was a value of 1.0).  Then I thought about it, and this is most likely due to floating point arithmetic.  Also, for the following reasons, I thought they were correct.  1)Based on LeNet architecture and 2)Validatin/Test accuracy were high 3) for the simpler images it guessed correctly 3) the ones that guessed incorrectly had a lot of noise and different colors in the background 4)it guessed correctly the simpler images with surprising accuracy

For the first image, the model is relatively sure that this is a right-of-way sign (probability of 1.0), and the image does contain a right-of-way. The top five soft max probabilities were
(11, 12, 26, 16, 21)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00     	| Right-of-way at the next intersection   		| 
| 2.64654807e-13   		| Priority Road  								|
| 9.86045453e-15		| Traffic signals								|
| 9.27697250e-15	   	| Vehicles over 3.5 metric tons prohibited		|
| 9.27697250e-15	    | Double curve      							|


For the second image, the model is relatively sure that this is a Priority road sign (probability of 1.0), and the image does contain a Priority road sign. The top five soft max probabilities were
(12, 13, 32, 41, 38)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Priority road  								| 
| 1.48634355e-10    	| Yield											|
| 1.58109268e-12		| End of all speed and passing limits			|
| 2.43505506e-14	    | End of no passing								|
| 4.49019407e-15		| Keep right-of-way 							|


For the third image, the model is relatively sure that this is a Priority road sign (probability of 0.99), but the image does not contain a Priority road sign. It has a No Entry sign.  I suspect this is because of the background colors and noise.  The top five soft max probabilities were
 (12, 20, 35,  9, 25)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.98599470e-01        | Priority road   								| 
| 1.39997201e-03     	| General caution								|
| 5.39363953e-07		| Ahead only									|
| 9.73968710e-08	   	| No passing							 		|
| 7.93840282e-09		| Road work      								|

For the fourth image, the model is relatively sure that this is a Road work sign (probability of 1.00), and the image does contain a Road work sign. The top five soft max probabilities were
(25, 14, 30, 24, 29)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Road work   									| 
| 1.05209830e-12    	| Stop 											|
| 8.60858134e-15		| Beware of ice/snow							|
| 7.66597160e-18	    | Road narrows on the right						|
| 9.73080900e-19	    | Bicycles crossing      						|

For the fith image, the model is relatively sure that this is a General caution sign (probability of 0.74), and the image does not contain a General caution sign. It has a Speed limit 70 km/h.   I suspect this is because of the background colors and noise.  The top five soft max probabilities were
 (25,  5,  3, 31, 18)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.43826509e-01        | Road Work   									| 
| 6.58523813e-02     	| Speed limit (80km/h) 							|
| 4.80511375e-02		| Speed limit (30km/h)							|
| 2.43359469e-02	    | Wild animals crossing							|
| 2.15997528e-02		| General caution 			    				|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
N/A

