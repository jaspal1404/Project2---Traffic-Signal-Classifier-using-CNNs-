# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architectur
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./exploratory_analysis.JPG "Data Visualization"
[image2]: ./grayscale.JPG "Grayscaling"
[image3]: ./web_images.JPG "Sign images from web"
[image4]: ./Softmax_probabilities_set1.JPG "Softmax_probabilities_Top3"
[image5]: ./Softmax_probabilities_set2.JPG "Softmax_probabilities_Top3"
[image6]: ./FeatureMap_conv1.JPG "Feature map Layer1"
[image7]: ./FeatureMap_pool1.JPG "Feature map Layer2"
[image8]: ./FeatureMap_conv2.JPG "Feature map Layer3"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Here are the steps I performed for preprocessing: 
* I decided to convert the images to grayscale because it is good to do so as mentioned in the LeNet implementation using CNNs and also suggested in the lessons as nice to have thing. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

* I normalized the image data because as explained in the lessons, the distribution of the input features is not uniform generally which makes training the model difficult, so it helps making the distribution uniform by bringing the mean of data around zero. 

I didn't use the data augmentation in first place (in the finalized architecture), but performed data augmentation in the iteration process although it couldn't help increase accuracy.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

After a lot of iterations, I trained the final model architecture using the below list of hyper parameters:

* Adam optimizer
* ReLu activation function
* learning rate 0.001
* batch size 128
* epochs 80
* Same architecture as LeNet with added different dropout values to all layers

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.92%
* validation set accuracy of 97.55 %
* test set accuracy of ?

I used an iterative approach to reach to the final model. Here is how I proceeded:

* What was the first architecture that was tried and why was it chosen?
Ans: I started with the same LeNet architecture described in the degree lessons. It was used to predict the signs data and gave good results, so thought to start with it as baseline (same hyper parameters).

* What were some problems with the initial architecture?
Ans: While training the model, I could only get to validation accuracy of 89% and more than 9% training accuracy, which clearly implied overfitting on the training data. Also I the images I was using were color images and the distribution of input data features was widely spread (without normalizing the data).

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Ans: I didn't changed the architecture in first place, but decided to tune some hyper parameters to improve the accuracy.

* Which parameters were tuned? How were they adjusted and why?
Ans: This is how I started:
    1. Firstly, applied grayscaling to the images and normalized the input data and trained for 100 epochs. On training, this improved the validation accuracy to 93.5% (all other parameters same as LeNet). But still the model was over fitted to training data (accuracy almost 99.99%).
    2. Secondly, added dropout regularization in only first 2 fully connected layers (fc1, fc2) with a keep_prob value of 0.7. This improved the validation accuracy further to 96.83%, thus reduced some over fitting.
    3. Further played around with multiple combinations of dropout values, thats when I finally arrived at the combination which gave me best validation accuracy with LeNet architecture 97.55% (keep_prob values of 0.9, 0.8, 0.7, 0.7 for conv1, conv2, fc1, fc2 respectively).
    4. Also used rate decay in my traing, to speed up the training process after certain training loss (used 0.001 and 0.0007 as two learning rates, although tried with different small values).

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Ans: As mentioned earlier, since LeNet architecture was a good baseline to start with, I kept playing around using same design but different combinations of hyper parameters. Although I got best accuracy with LeNet design, still I tried few changes in design:
    1. Removed one fully connected layer and trained the model.
    2. Removed two fully connected layers and trained the model.
    3. Also added one more conv layer (without pooling) after removing 2 fully connected layers.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image3]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way  		| Right-of-way   									| 
| Speed limit (60km/h)  			| Speed limit (60km/h)  										|
| Speed limit (30km/h)  					| Speed limit (50km/h)  											|
| Priority Road 			| Priority Road   					 				|
| Keep Right  			| Keep Right    							|
| Turn left ahead   			| Turn left ahead    							|
| General caution   			| General caution    							|
| Road work   			| Road work     							|


The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This compares favorably to the accuracy on the test set of 94.54%.
The only image that model was unable to predict correctly was: Speed limit (30km/h), that was predicted as: Speed limit (50km/h).
One reason I could see is the low data availability for that particular class, as you can see in the visualization above class 1 only has around 200 images in the training dataset due to which model is unable to perform very well on the test image (although it was close predicting it as a 2nd guess).


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here is the visualization of the softmax probabilities of the images downloaded from web, along with the first 3 guess of the model on that image.

![alt text][image4]
![alt text][image5]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I would like to say that I enjoyed a lot during this project and learnt as well. First tine I did worked with Nueral networks in so detail and it was fun doing a deep dive and playing around with things. I think actually knowing and visulaizing what layers in the network are doing to learn and predict the images is really awesome and interesting.

So I viewed the feature maps for couple of web images and was able to know more, how the initial layers start learning about the edges in the image and further layers start looking at more complex features outlining the same in terms of intensity of pixels etc. Here is the output features for one of the images for first 3 layers (conv1, pool1, conv2).

![alt text][image6]
![alt text][image7]
![alt text][image8]
