
# **Traffic Sign Recognition** 

## Writeup 

### Jack Zhang 11/7/2017
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

[image1]: ./images/vis_orig.png "Sampling of original test images"
[image2]: ./images/histogram.png "Histogram"
[image3]: ./images/grayscale.png "Grayscale images"
[image4]: ./images/normalized.png "Normalized images"
[image5]: ./images/signs.png "German signs from internet"
[image6]: ./images/sign_probs.png "Sign probabilities"
[image7]: ./images/sign_indices.png "Sign indices"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jyzhang23/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a random sampling of 27 images from the original training set. There are a number of different lighting conditions, with some signs having little contrast due to over or under saturation.

![Sampling of original training set][image1]

In addition, here are 3 histograms of the training, validation, and test data sets, binned by class ID. They show relatively similar distribution between the 3 data sets, although the distribution of signs within each data set is not even.

![Histograms][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it would make the next step, normalization, simpler, and also from this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the authors achieved slightly better performance using grayscale rather than color.

I experimented with different color channel balances in the grayscale conversion, but ultimately used the standard conversion from openCV. Here is an example of random sampled images from the training data set after converting to grayscale.

![Grayscale samples][image3]

As a last step, I normalized the image data to have mean of zero and variance of one within each image in order to prevent the weights from oversaturating and to help with the learning process. Here I also tried out a few different normalization approaches. In the end, I found the mean and standard deviation in each image, and subtracted the mean from the image and divided by the standard deviation. 

Here is a random sample of 27 training set images after normalization

![Normalized images][image4]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|Input Size|Output Size| 
|:---------------------:|:---------------------------------------------:|:--------:|:--------:| 
| Input         		| grayscale image   					        |          |32x32x1   |
| Convolution 5x5     	| 1x1 stride, valid padding                 	|32x32x1   |28x28x20  |
| RELU					|												|          |          |
| Max pooling	      	| 2x2 stride, valid padding                   	|28x28x20  |14x14x20  |
| Convolution 5x5	    | 1x1 stride, valid padding                  	|14x14x20  |10x10x40  |
| RELU	                |           									|          |          |
| Max pooling	      	| 2x2 stride, valid padding                     |10x10x40  |5x5x40    |
| Fully connected		|            									|1000      |400       |
| Dropout       		| 25% keep prob									|          |          |
| RELU	                |           									|          |          |
| Fully connected		|           									|400       |120       |
| Dropout       		| 25% keep prob									|          |          |
| RELU	                |           									|          |          |
| Fully connected		|            									|120       |43        |
| Softmax				| final probabilities                           |          |          ||
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a cross entropy cost function and Adam optimizer, which has some nice features such as momentum and natural learning rate adaptation. Because my computer is relatively old, this is really beneficial in that it allows less hyperparameter tuning.

I used a batch size of 128 and learning rate of 0.001. After 10 epochs, it had not quite finished training, so I ran 15 additional epochs with a learning rate of 0.0005. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.8%
* validation set accuracy of 94.9% 
* test set accuracy of 94.1%

I started by using the LeNet architecture from the lab. This model was successfully used for classifying numerical digits in the MNIST data set, which is similar to what I wanted to accomplish in this lab. Also, a [similar model](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) was used to classify traffic signs.

The initial model resulted in validation accuracies of 75-80% after a few epochs, but then quickly fell down to below 10% with additional training steps. To me, this seemed like overfitting. To reduce overfitting, I added dropout layers to my model. I experimented with how many dropout layers were implemented, as well as where they were placed, and what percentage to keep in the dropout layers. 

I also increased the depth of my convolution and fully connected layers, since this project involves over 4 times more classifications than the lab. I also experimented with the image normalization, using 0-255 image scaling, and different grayscale adjustments. These pre-processing steps did not have a significant impact, and actually sometimes made the model a little worse. 

In the end, the model finally achieved good validation accuracies with the increased layer depths and a dropout keep rate of 25%. This suggests that the original model suffered from low accuracy due to overfitting.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eleven German traffic signs that I found on the web:

![German signs downloaded from internet][image5] 

I thought the second image might be difficult to classify because it's a little off-center, and there's half of another sign in the image. The others I expected the classifier to be successful predicting.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way      	| Right-of-way   								| 
| Gneral caution     	| 30km/h 							            |
| Stop					| Stop											|
| Yield	      		    | Yield					 				        |
| 50km/h			    | 60km/h      							        |
| 30km/h      	        | Roundabout mandatory   						|
| 20km/h      	        | Stop   								        | 
| 100km/h     	        | 100km/h 							            |
| Priority				| Priority										|
| No passing	        | No passing					 				|
| No entry			    | No entry      							    ||

The model was able to correctly guess 7 of the 11 traffic signs, which gives an accuracy of 64%. This is lower than expected, considering the training, validation, and test accuracies were all over 94%. The image I thought might have trouble, the second one, did indeed predict the wrong sign. What's surprising was that the speed limit signs had difficulties being classified correctly. This may be due to the stronger contrast in the images I found from the internet than the training data. Also, the perspective of these signs are from a lower perspective.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is as follows

```python
with tf.Session() as sess:
    saver.restore(sess, './lenet2')
    
    sign_labels=tf.one_hot(sign_class, 43)
    softmax_probs = tf.nn.softmax(logits)
    top_5=tf.nn.top_k(softmax_probs,5)
    probs=sess.run(top_5, feed_dict={x: images_norm, keep_prob: 1.0})
    print(probs)
    probs_arry=probs
```
For each image, the top five predicted probabilities are shown above, while the figure below shows the corresponding sign class IDs.

![Predicted probabilities][image6] 
![Predicted indices][image7] 

For the images that were correctly predicted, the probabilities are very high, except for the 100km/h sign (42%). As mentioned earlier, the downloaded speed limit signs had difficulties with the classification, so even the one that was correctly classified showed a lower probability than usual.

Looking at the incorrectly classified images, the correct classification of image 2 (class ID=18) does not appear in the top 5 probabilities, and the probabilities are all relatively low and even, which suggest the model was not sure how to classify this image.

In contrast, for the incorrectly classified image 5, the classifier had a 99% certainty it was either one of two classes, neither of which was true. The actual class ID (3), appears as the third most probable, which has lower than 1% probability.

This is also the case for image 7; the actual class ID (0) is the third highest probability, 10% in this case.

For image 6, the correct class ID, 1, does not appear in the top 5 probabilities.

It is clear that these speed limit signs are not well classified, which might be due to the different contrast and perspective.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


