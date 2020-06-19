# Machine_Learning (ML)
Machine Learning is the field of study that gives computers the ability to learn without  being explicitly programmed.<br>

ML is divided into two categories:<br>
-Supervised <br>
-Unsupervised<br>

Supervised is where data is labelled and the program learns to predict the output from the input data<br>

This can be broken into: <br>
-Regression: problems here include prediction of a continous valued output e.g housing prices in Nairobi. <br>
-Classification: problems here deal with prediction of discrete number of values e.g is a particlar email spam or not <br>

Unsupervised Learning is where the program learns the inherent structure of the data based on unlabeled examples.<br>

Clustering is a common unsupervised machine learning approach that finds patterns and structures in unlabeled data by grouping them into clusters. e.g Search engines to group similar objects in one cluster <br>

#### Differences.
Supervised Learning: data is labeled and the program learns to predict the output from the input data.<br>
Unsupervised Learning: data is unlabeled and the program learns to recognize the inherent structure in the input data. <br>

## The Process.
The process of performing Machine Learning often requires many more steps before and after the predictive analytics.<br>
We try to think of the Machine Learning process as:<br>

#### 1.Formulating a Question.<br>
What is it that we want to find out? How will we reach the success criteria that we set?<br>
When we’re thinking about creating a model, we have to narrow down to one measurable, specific task. For example, we might say we want to predict the wait times for customers’ food orders within 2 minutes, so that we can give them an accurate time estimate.

#### 2.Finding and Understanding the Data.<br>
The largest chunk of time in any machine learning process is finding the relevant data to help answer your question, and getting it into the format necessary for performing predictive analysis.<br>
Once you have your data, you want to understand it so that you will know what model to apply and what the outputs will mean. First, you will want to examine the summary statistics:<br>
   -Calculate means and medians to understand the distribution<br>
   -Calculate percentiles.<br>
   -Find correlations that indicate relationships.<br>
You may also want to visualize the data, perhaps using box plots to identify outliers, histograms to show the basic structure of the data, and scatter plots to examine relationships between variables.<br>

#### 3.Cleaning the Data and Feature Engineering.<br>
Real data is messy! Data may have errors. Some columns may be empty. The features we’re interested in might require string manipulation to extract. Cleaning the data refers to the process by which we address missing values and outliers, among other things that may affect our insights.<br>
Feature Engineering refers to the process by which we choose the important features (or columns) to look at, and make the appropriate transformations to prepare our data for our model.<br>
We might try:<br>
   -Normalizing or standardizing the data.<br>
   -Augmenting the data by adding new columns.<br>
   -Removing unnecessary columns.<br>
After we test our model on the data we have, we might go back and reengineer features to see if we get a better result.<br>

#### 4.Choosing a Model.<br>
Once we understand our dataset and know the problem we are trying to solve, we can begin to choose a model that will help us tackle our problem.<br>
If we are attempting to find a continuous output, like predicting the number of minutes someone should wait for their order, we would use a regression algorithm.<br>
If we are attempting to classify an input, like determining if an order will take under 5 minutes or over 10 mins, then we would use a classification algorithm.<br>

#### 5.Tuning and Evaluating.<br>
Each model has a variety of parameters that change how it makes decisions. We can adjust these and compare the chosen evaluation metrics of the different variants to find the most accurate model.<br>

#### 6.Using the Model and Presenting Results.<br>
When you achieve the level of accuracy you want on your training set, you can use the model on the data you actually care about analyzing.<br>
For our example, we can now start inputting new orders. The input could be an order, with features like:<br>
   -the type of item ordered.<br>
   -the quantity.<br>
   -the time of day.<br>
   -the number of employees working.<br>
The output would be how long the order is expected to take. This information could be displayed to users.
