# Machine_Learning (ML)
Machine Learning is the field of study that gives computers the ability to learn without  being explicitly programmed.<br>

ML is divided into two categories:<br>
-Supervised <br>
-Unsupervised<br>

Supervised is where data is labelled and the program learns to predict the output from the input data<br>

This can be broken into: <br>
-Regression: problems here include prediction of a continous valued output e.g housing prices in Nairobi. <br>
-Classification: problems here deal with prediction of discrete number of values e.g is a particlar email spam or not <br>

#### Regression Vs Classification
Regression is used to predict outputs that are continuous. 
The outputs are quantities that can be flexibly determined based on the inputs of the model rather than being confined to a set of possible labels.<br>
For example: Predict the height of a potted plant from the amount of rainfall<br>

Classification
Classification is used to predict a discrete label.
The outputs fall under a finite set of possible outcomes. 
Many situations have only two possible outcomes. This is called binary classification(True/False).<br>
For example: Predict whether it will rain or not<br>

Multi-label classification is when there are multiple possible outcomes. 
It is useful for customer segmentation, image categorization, and sentiment analysis for understanding text. 
To perform these classifications, we use models like Naive Bayes, K-Nearest Neighbors, and SVMs.

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

## Scikit-Learn.
Scikit-learn is a library in Python that provides many unsupervised and supervised learning algorithms. It’s built upon some of the technologies like NumPy, pandas, and Matplotlib! <br>

The functionality that scikit-learn provides include:
    -Regression, including Linear and Logistic Regression.<br>
    -Classification, including K-Nearest Neighbors.<br>
    -Clustering, including K-Means and K-Means++ .<br>
    -Model selection.<br>
    -Preprocessing, including Min-Max Normalization.<br>
    
### Scikit-Learn Cheatsheet.
#### Linear Regression.<br>
Import and create the model: <br>

```python 
from sklearn.linear_model import LinearRegression.

your_model = LinearRegression().
```

fit: <br>

```python
your_model.fit(x_training_data, y_training_data)
```
.coef_: contains the coefficients<br>
.intercept_: contains the intercept<br>

Predict:<br>

``` python
predictions = your_model.predict(your_x_data)
```
.score(): returns the coefficient of determination R².

#### Naive Bayes
Import and create the model: <br>

```python
from sklearn.naive_bayes import MultinomialNB

your_model = MultinomialNB()
```
Fit:<br>

```python
your_model.fit(x_training_data, y_training_data)
```

Predict:<br>

``` python
# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)
```
```python
# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)

```

#### K-Nearest Neighbors
Import and create the model:<br>

```python
from sklearn.neigbors import KNeighborsClassifier

your_model = KNeighborsClassifier()
```
Fit:<br>

```python
your_model.fit(x_training_data, y_training_data)
```
Predict:<br>

```python
# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)
```
```python
# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)
```

#### K-Means
Import and create the model:<br>

```python
from sklearn.cluster import KMeans

your_model = KMeans(n_clusters=4, init='random')
```
n_clusters: number of clusters to form and number of centroids to generate<br>

init: method for initialization<br>

k-means++: K-Means++ [default] <br>

random: K-Means<br>

random_state: the seed used by the random number generator [optional]<br>

Fit:<br>

```python
your_model.fit(x_training_data)
```

Predict:<br>

```python
predictions = your_model.predict(your_x_data)
```
#### Validating the Model
Import and print accuracy, recall, precision, and F1 score:<br>

```python
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print(accuracy_score(true_labels, guesses))

print(recall_score(true_labels, guesses))

print(precision_score(true_labels, guesses))

print(f1_score(true_labels, guesses))
``` 

Import and print the confusion matrix:<br>

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(true_labels, guesses))
```
#### Training Sets and Test Sets 
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
```
train_size: the proportion of the dataset to include in the train split<br>

test_size: the proportion of the dataset to include in the test split <br>

random_state: the seed used by the random number generator [optional] <br>

### Linear Regression
Representing Points, distance.<br>
Three different ways to define the distance between two points:<br>
-Euclidean Distance<br>
Euclidean Distance is the most commonly used distance formula. 
To find the Euclidean distance between two points, we first calculate the squared distance between each dimension. 
If we add up all of these squared differences and take the square root, we’ve computed the Euclidean distance.
 
-Manhattan Distance<br>
Manhattan Distance is extremely similar to Euclidean distance. 
Rather than summing the squared difference between each dimension, 
we instead sum the absolute value of the difference between each dimension

-Hamming Distance<br>
Instead of finding the difference of each dimension, 
Hamming distance only cares about whether the dimensions are exactly equal. 
When finding the Hamming distance between two points, add one for every dimension that has different values.
Hamming distance is used in spell checking algorithms.<br>

SciPy Distances.<br>
Implementation of the above distances using scipy python library.<br>
-Euclidean Distance .euclidean()<br>
-Manhattan Distance .cityblock()<br>
-Hamming Distance .hamming()<br>
scipy implementation of Hamming distance will always return a number between 0 an 1.

## Data Manipulation with Numpy
### Numpy Arrays
a. Numpy Arrays <br>
NumPy arrays are basically just Python lists with added features. You can easily convert a Python list to a Numpy array using the np.array function which takes in a Python list as its required argument. The function has quite a few keyword arguments, but the main one to know is dtype. 
The dtype keyword argument takes in a NumPy type and manually casts the array to the specified type.<br>
Example:<br>
The code below is an example usage of np.array to create a 2-D matrix. Note that the array is manually cast to np.float32.<br>

```python 
import numpy as np

arr = np.array([[0, 1, 2], [3, 4, 5]],
               dtype=np.float32)
print(repr(arr))

#output array([[0., 1., 2.], [3., 4., 5.]], dtype=float32)
```

When the elements of a NumPy array are mixed types, then the array's type will be upcast to the highest level type. Meaning that if an array input has mixed int and float elements, all the integers will be cast to their floating-point equivalents. 
If an array is mixed with int, float, and string elements, everything is cast to strings.<br>
Example of np.array upcasting. Both integers are cast to their floating-point equivalents.<br>
```python
import numpy as np

arr = np.array([0, 0.1, 2])
print(repr(arr))

#output array([0. , 0.1, 2. ])
```

b. Copying<br>
Similar to Python lists, when we make a reference to a NumPy array it doesn't create a different array. Therefore, if we change a value using the reference variable, it changes the original array as well. We get around this by using an array's inherent copy function. The function has no required arguments, and it returns the copied array.<br>
Example below, c is a reference to a while d is a copy. Therefore, changing c leads to the same change in a, while changing d does not change the value of b.<br>
```python
import numpy as np
a = np.array([0, 1])
b = np.array([9, 8])
c = a
print('Array a: {}'.format(repr(a)))
c[0] = 5
print('Array a: {}'.format(repr(a)))

d = b.copy()
d[0] = 6
print('Array b: {}'.format(repr(b)))

#output Array a: array([0, 1]),Array a: array([5, 1]),Array b: array([9, 8])
```

c. Casting<br>
We cast NumPy arrays through their inherent astype function. The function's required argument is the new type for the array.
It returns the array cast to the new type.<br>
The  example below is on casting using the astype function. The dtype property returns the type of an array.<br>
```python
arr = np.array([0, 1, 2])
print(arr.dtype)
arr = arr.astype(np.float32)
print(arr.dtype)

#output int64 float32
```
d. NaN<br>
When we don't want a NumPy array to contain a value at a particular index, we can use np.nan to act as a placeholder.
A common usage for np.nan is as a filler value for incomplete data.<br>
Example usage of np.nan. Note that np.nan cannot take on an integer type.<br>
```python
arr = np.array([np.nan, 1, 2])
print(repr(arr))

arr = np.array([np.nan, 'abc'])
print(repr(arr))

# Will result in a ValueError
np.array([np.nan, 1, 2], dtype=np.int32)

#other ouput array([nan,  1.,  2.]) array(['nan', 'abc'], dtype='<U32')
```

e. Infinity<br>
To represent infinity in NumPy, we use the np.inf special value. We can also represent negative infinity with -np.inf.<br>
Example usage of np.inf. Note that np.inf cannot take on an integer type.<br>

```python
print(np.inf > 1000000)

arr = np.array([np.inf, 5])
print(repr(arr))

arr = np.array([-np.inf, 1])
print(repr(arr))

# Will result in an OverflowError
np.array([np.inf, 3], dtype=np.int32)

#other output True, array([inf,  5.]), array([-inf,   1.])
```
More illustrations of the above can be found [here](https://github.com/veldakarimi/Machine_Learning/blob/ca21216a2b73282fe7ac71e3f3c7a0bce738be2a/numpyarrays.py#L1-L14)<br>
### Numpy Basics
Perform basic operations to create and modify NumPy arrays.
a. Ranged Data<br>
np.array  is used to create an array which is equal to hardcoding.<br>
NumPy provides an option to create ranged data arrays using np.arange. 
The function acts very similar to the range function in Python, and will always return a 1-D array.
Example:<br>
```python
arr = np.arange(5)
print(repr(arr))

arr = np.arange(5.1)
print(repr(arr))

arr = np.arange(-1, 4)
print(repr(arr))

arr = np.arange(-1.5, 4, 2)
print(repr(arr))

#output
#array([0, 1, 2, 3, 4])
#array([0., 1., 2., 3., 4., 5.])
#array([-1,  0,  1,  2,  3])
#array([-1.5,  0.5,  2.5])
```
The output of np.arange is specified as follows:<br>
-If only a single number, n, is passed in as an argument, np.arange will return an array with all the integers in the range [0,n] Note: the lower end is inclusive while the upper end is exclusive.<br>
-For two arguments, m and n, np.arange will return an array with all the integers in the range [m, n].<br>
-For three arguments, m, n, and s, np.arange will return an array with the integers in the range [m, n] using a step size of s.<br>
-Like np.array, np.arange performs upcasting. It also has the dtype keyword argument to manually cast the array.<br>

To specify the number of elements in the returned array, rather than the step size, we can use the np.linspace function.<br>
This function takes in a required first two arguments, for the start and end of the range, respectively.<br>
The end of the range is inclusive for np.linspace, unless the keyword argument endpoint is set to False.<br>
To specify the number of elements, we set the num keyword argument (its default value is 50).
Example:<br>
```python
arr = np.linspace(5, 11, num=4)
print(repr(arr))

arr = np.linspace(5, 11, num=4, endpoint=False)
print(repr(arr))

arr = np.linspace(5, 11, num=4, dtype=np.int32)
print(repr(arr))

#output
#array([ 5.,  7.,  9., 11.])
#array([5. , 6.5, 8. , 9.5])
#array([ 5,  7,  9, 11], dtype=int32)
```
b. Reshaping data<br>
We use np.reshape to reshape data.<br>
It takes in an array and a new shape as required arguments. The new shape must exactly contain all the elements from the input array. E.g, we could reshape an array with 12 elements to (4, 3), but we can't reshape it to (4, 4).<br>
We are allowed to use the special value of -1 in at most one dimension of the new shape. The dimension with -1 will take on the value necessary to allow the new shape to contain all the elements of the array.
Example:<br>
```python
arr = np.arange(8)

reshaped_arr = np.reshape(arr, (2, 4))
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))

reshaped_arr = np.reshape(arr, (-1, 2, 2))
print(repr(reshaped_arr))
print('New shape: {}'.format(reshaped_arr.shape))
#output array([[0, 1, 2, 3], [4, 5, 6, 7]]) New shape: (2, 4) array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]) New shape: (2, 2, 2)
```
While the np.reshape function can perform any reshaping utilities we need, NumPy provides an inherent function for flattening an array.<br>
Flattening an array reshapes it into a 1D array. Since we need to flatten data quite often, it is a useful function.<br>
Example:<br>
```python
arr = np.arange(8)
arr = np.reshape(arr, (2, 4))
flattened = arr.flatten()
print(repr(arr))
print('arr shape: {}'.format(arr.shape))
print(repr(flattened))
print('flattened shape: {}'.format(flattened.shape))

#output 
#array([[0, 1, 2, 3],
       #[4, 5, 6, 7]])
#arr shape: (2, 4)
#array([0, 1, 2, 3, 4, 5, 6, 7])
#flattened shape: (8,)
```
c. Transposing<br>
The general meaning of transposing is to change places or context.<br>
Perharps we have data that's supposed to be in a particular format, but some new data we get is rearranged. We can transpose the data, using the np.transpose function, to convert it to the proper format.<br>
The code below shows an example usage of the np.transpose function. The matrix rows become columns after the transpose.
```python
arr = np.arange(8)
arr = np.reshape(arr, (4, 2))
transposed = np.transpose(arr)
print(repr(arr))
print('arr shape: {}'.format(arr.shape))
print(repr(transposed))
print('transposed shape: {}'.format(transposed.shape))

#output
#array([[0, 1],
       #[2, 3],
       #[4, 5],
       #[6, 7]])
#arr shape: (4, 2)
#array([[0, 2, 4, 6],
       #[1, 3, 5, 7]])
#transposed shape: (2, 4)
```
The function takes in a required first argument, which will be the array we want to transpose. It also has a single keyword argument called <b>axes</b>, which represents the new permutation of the dimensions.<br>
The permutation <i>is a tuple/list of integers, with the same length as the number of dimensions in the array</i>. It tells us where to switch up the dimensions. For example, if the permutation had 3 at index 1, it means the old third dimension of the data becomes the new second dimension (since index 1 represents the second dimension).<br>
The code below shows an example usage of the np.transpose function with the axes keyword argument. The shape property gives us the shape of an array.<br>
```python
arr = np.arange(24)
arr = np.reshape(arr, (3, 4, 2))
transposed = np.transpose(arr, axes=(1, 2, 0))
print('arr shape: {}'.format(arr.shape))
print('transposed shape: {}'.format(transposed.shape))

#output
#arr shape: (3, 4, 2)
#transposed shape: (4, 2, 3)
```
In this example, the old first dimension became the new third dimension, the old second dimension became the new first dimension, and the old third dimension became the new second dimension. The default value for axes is a dimension reversal (e.g. for 3-D data the default axes value is [2, 1, 0]).

d.Zeros and ones
Sometimes, we need to create arrays filled solely with 0 or 1.e.g binary data, we may need to create dummy datasets of strictly one label. For creating these arrays, NumPy provides the functions np.zeros and np.ones. They both take in the same arguments, which includes just one required argument, the array shape.<br>
The functions also allow for manual casting using the dtype keyword argument.<br>
The code below shows example usages of np.zeros and np.ones.<br>
```python
arr = np.zeros(4)
print(repr(arr))

arr = np.ones((2, 3))
print(repr(arr))

arr = np.ones((2, 3), dtype=np.int32)
print(repr(arr))

#output
#array([0., 0., 0., 0.])
#array([[1., 1., 1.],
       #[1., 1., 1.]])
#array([[1, 1, 1],
       #[1, 1, 1]], dtype=int32)
```
If we want to create an array of 0's or 1's with the same shape as another array, we can use np.zeros_like and np.ones_like.<br>
The code below shows example usages of np.zeros_like and np.ones_like.<br>
```python
arr = np.array([[1, 2], [3, 4]])
print(repr(np.zeros_like(arr)))

arr = np.array([[0., 1.], [1.2, 4.]])
print(repr(np.ones_like(arr)))
print(repr(np.ones_like(arr, dtype=np.int32)))
#output
#array([[0, 0],
       #[0, 0]])
#array([[1., 1.],
       #[1., 1.]])
#array([[1, 1],
       #[1, 1]], dtype=int32)
```





